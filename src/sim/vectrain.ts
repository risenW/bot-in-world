// Vectorized rollout + PPO training loop over NavEnvs.
// Runs identically in a web worker (live training UI) and Node (pretraining).

import { NavGrid } from './navgrid';
import { NavEnv, OBS_SIZE, N_ACTIONS, TaskConfig, DEFAULT_TASK } from './env';
import {
  Policy, PpoTrainer, PpoConfig, DEFAULT_PPO, Rollout, makeRollout,
  makeActivations, Activations, computeGae, softmaxRow, sampleCategorical, UpdateStats,
} from './ppo';
import { mulberry32, Rng } from './rng';

export interface TrainStats {
  globalStep: number;
  update: number;
  sps: number;
  meanReturn: number;     // rolling mean episodic return
  successRate: number;    // rolling
  meanEpLen: number;
  curriculum: number;     // nav: max goal distance (m); fetch: active ball count
  losses: UpdateStats;
  lr: number;
}

// Auto-curriculum on goal distance: expand when the bot succeeds reliably.
const CURRICULUM_MIN = 4.0;
const CURRICULUM_MAX = 14.0;
const CURRICULUM_STEP = 0.5;

export class VecTrainer {
  envs: NavEnv[];
  policy: Policy;
  trainer: PpoTrainer;
  cfg: PpoConfig;
  rng: Rng;
  roll: Rollout;
  obsBuf: Float32Array;       // current obs per env
  acts: Activations;          // batch = numEnvs (rollout-time)
  probs: Float32Array;
  globalStep = 0;
  update = 0;

  // rolling episode stats
  private epReturns: Float32Array;
  private epLens: Int32Array;
  private recent: { ret: number; len: number; success: boolean }[] = [];
  curriculumMax = CURRICULUM_MIN;

  task: TaskConfig;
  ballCurriculum = 1;

  constructor(grid: NavGrid, seed = 42, cfgOverride: Partial<PpoConfig> = {}, task: TaskConfig = DEFAULT_TASK) {
    this.cfg = { ...DEFAULT_PPO, obsSize: OBS_SIZE, actions: N_ACTIONS, ...cfgOverride };
    this.task = { ...task };
    this.rng = mulberry32(seed);
    this.policy = new Policy(OBS_SIZE, this.cfg.hidden, N_ACTIONS, this.rng);
    this.trainer = new PpoTrainer(this.policy, this.cfg, this.rng);
    this.envs = [];
    for (let e = 0; e < this.cfg.numEnvs; e++) {
      const env = new NavEnv(grid, mulberry32(seed * 7919 + e), this.task);
      this.applyCurriculum(env);
      env.reset();
      this.envs.push(env);
    }
    const steps = this.cfg.numEnvs * this.cfg.horizon;
    this.roll = makeRollout(steps, OBS_SIZE);
    this.obsBuf = new Float32Array(this.cfg.numEnvs * OBS_SIZE);
    this.acts = makeActivations(this.policy, this.cfg.numEnvs);
    this.probs = new Float32Array(this.cfg.numEnvs * N_ACTIONS);
    this.epReturns = new Float32Array(this.cfg.numEnvs);
    this.epLens = new Int32Array(this.cfg.numEnvs);
    for (let e = 0; e < this.cfg.numEnvs; e++) this.envs[e].observe(this.obsBuf, e * OBS_SIZE);
  }

  private applyCurriculum(env: NavEnv): void {
    env.goalDistMin = 1.0;
    env.goalDistMax = this.curriculumMax;
    if (this.task.mode === 'fetch') {
      env.activeBalls = Math.max(1, Math.min(this.ballCurriculum, this.task.numBalls));
    }
  }

  loadWeights(flat: Float32Array): void {
    this.policy.load(flat);
  }

  // One PPO iteration: collect horizon steps from all envs, then update.
  iterate(lrScale = 1): TrainStats {
    const { cfg, envs, roll } = this;
    const N = cfg.numEnvs, H = cfg.horizon, O = OBS_SIZE, A = N_ACTIONS;
    const t0 = nowMs();

    for (let t = 0; t < H; t++) {
      this.policy.forward(this.obsBuf, N, this.acts);
      for (let e = 0; e < N; e++) {
        const i = t * N + e;
        softmaxRow(this.acts.logits, e * A, A, this.probs, e * A);
        const action = sampleCategorical(this.probs, e * A, A, this.rng());
        const logprob = Math.log(Math.max(this.probs[e * A + action], 1e-10));

        roll.obs.set(this.obsBuf.subarray(e * O, e * O + O), i * O);
        roll.actions[i] = action;
        roll.logprobs[i] = logprob;
        roll.values[i] = this.acts.values[e];

        const env = envs[e];
        const res = env.step(action);
        roll.rewards[i] = res.reward;
        roll.dones[i] = res.done ? 1 : 0;

        this.epReturns[e] += res.reward;
        this.epLens[e]++;
        if (res.done) {
          this.pushEpisode(this.epReturns[e], this.epLens[e], res.success);
          this.epReturns[e] = 0;
          this.epLens[e] = 0;
          this.applyCurriculum(env);
          env.reset();
        }
        env.observe(this.obsBuf, e * O);
      }
    }
    this.globalStep += N * H;

    // bootstrap values for the obs after the last step
    this.policy.forward(this.obsBuf, N, this.acts);
    computeGae(roll, this.acts.values, N, H, cfg.gamma, cfg.gaeLambda);

    const lr = cfg.lr * lrScale;
    const losses = this.trainer.update(roll, lr);
    this.update++;

    // snapshot stats BEFORE curriculum adjustment (it clears the rolling buffer)
    const sr = this.successRate();
    const meanRet = this.meanReturn();
    const meanLen = this.meanEpLen();
    if (this.recent.length >= 100) {
      if (this.task.mode === 'fetch') {
        // ramp ball count first, then goal distance
        if (sr > 0.7 && this.ballCurriculum < this.task.numBalls) {
          this.ballCurriculum++;
          this.recent.length = 0;
        } else if (sr > 0.8 && this.curriculumMax < CURRICULUM_MAX) {
          this.curriculumMax += CURRICULUM_STEP;
          this.recent.length = 0;
        } else if (sr < 0.1 && this.ballCurriculum > 1) {
          this.ballCurriculum--;
          this.recent.length = 0;
        }
      } else if (sr > 0.8 && this.curriculumMax < CURRICULUM_MAX) {
        this.curriculumMax += CURRICULUM_STEP;
        this.recent.length = 0;
      } else if (sr < 0.15 && this.curriculumMax > CURRICULUM_MIN) {
        this.curriculumMax -= CURRICULUM_STEP;
        this.recent.length = 0;
      }
    }

    const dt = (nowMs() - t0) / 1000;
    return {
      globalStep: this.globalStep,
      update: this.update,
      sps: Math.round((N * H) / Math.max(dt, 1e-6)),
      meanReturn: meanRet,
      successRate: sr,
      meanEpLen: meanLen,
      curriculum: this.task.mode === 'fetch' ? this.ballCurriculum : this.curriculumMax,
      losses,
      lr,
    };
  }

  private pushEpisode(ret: number, len: number, success: boolean): void {
    this.recent.push({ ret, len, success });
    if (this.recent.length > 200) this.recent.shift();
  }
  meanReturn(): number {
    if (!this.recent.length) return 0;
    return this.recent.reduce((s, e) => s + e.ret, 0) / this.recent.length;
  }
  successRate(): number {
    if (!this.recent.length) return 0;
    return this.recent.filter((e) => e.success).length / this.recent.length;
  }
  meanEpLen(): number {
    if (!this.recent.length) return 0;
    return this.recent.reduce((s, e) => s + e.len, 0) / this.recent.length;
  }
}

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
