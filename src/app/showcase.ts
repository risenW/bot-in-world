// Showcase: a live NavEnv on the main thread driven by the latest policy
// weights — this is the bot you watch while (or after) training.

import * as pc from 'playcanvas';
import { NavEnv, OBS_SIZE, N_ACTIONS, DT } from '../sim/env';
import { NavGrid, groundHeight, isWalkable, castRay } from '../sim/navgrid';
import { Policy, makeActivations, Activations, softmaxRow, sampleCategorical, argmaxRow } from '../sim/ppo';
import { mulberry32, Rng } from '../sim/rng';
import { HumanoidBot, GoalMarker } from './bot';

export class Showcase {
  env: NavEnv;
  policy: Policy;
  bot: HumanoidBot;
  goal: GoalMarker;
  grid: NavGrid;
  greedy = false;        // argmax vs sampled actions
  paused = false;

  episodes = 0;
  successes = 0;

  private acts: Activations;
  private probs = new Float32Array(N_ACTIONS);
  private obs = new Float32Array(OBS_SIZE);
  private rng: Rng;
  private accum = 0;
  private sinceDone = 0;

  onEpisodeEnd: ((success: boolean) => void) | null = null;

  constructor(app: pc.Application, grid: NavGrid) {
    this.grid = grid;
    this.rng = mulberry32(Math.floor(Math.random() * 1e9));
    this.env = new NavEnv(grid, this.rng);
    this.env.goalDistMin = 2;
    this.env.goalDistMax = 9;
    this.policy = new Policy(OBS_SIZE, 128, N_ACTIONS, this.rng);
    this.acts = makeActivations(this.policy, 1);
    this.bot = new HumanoidBot(app);
    this.goal = new GoalMarker(app);
    this.respawn();
  }

  setGrid(grid: NavGrid): void {
    this.grid = grid;
    this.env = new NavEnv(grid, this.rng);
    this.env.goalDistMin = 2;
    this.env.goalDistMax = 9;
    this.episodes = 0; this.successes = 0;
    this.respawn();
  }

  setWeights(flat: Float32Array): void {
    if (flat.length !== this.policy.serialize().length) {
      console.warn('showcase: weight size mismatch, ignoring');
      return;
    }
    this.policy.load(flat);
  }

  setPolicy(policy: Policy): void {
    this.policy = policy;
    this.acts = makeActivations(policy, 1);
  }

  clickGoal(x: number, z: number): boolean {
    if (!isWalkable(this.grid, x, z)) return false;
    this.env.setGoal(x, z);
    this.syncGoal();
    return true;
  }

  respawn(): void {
    // prefer open spots so the bot (and camera) aren't wedged against geometry
    for (let attempt = 0; attempt < 30; attempt++) {
      this.env.reset();
      let minClear = Infinity;
      for (let i = 0; i < 8; i++) {
        const ang = (i / 8) * Math.PI * 2;
        minClear = Math.min(minClear, castRay(this.grid, this.env.x, this.env.z, Math.sin(ang), Math.cos(ang), 2));
      }
      if (minClear > 0.9) break;
    }
    this.bot.snapTo(this.env.x, this.env.z, this.env.yaw);
    this.syncGoal();
  }

  private syncGoal(): void {
    this.goal.setPosition(this.env.goalX, groundHeight(this.grid, this.env.goalX, this.env.goalZ), this.env.goalZ);
  }

  update(dt: number): void {
    this.goal.update(dt);
    if (!this.paused) {
      this.accum += dt;
      while (this.accum >= DT) {
        this.accum -= DT;
        this.stepSim();
      }
    }
    const alpha = Math.min(1, this.accum / DT);
    const gy = groundHeight(this.grid, this.bot.position.x, this.bot.position.z);
    this.bot.render(alpha, Number.isNaN(gy) ? 0 : gy, dt);
  }

  private stepSim(): void {
    if (this.sinceDone > 0) { this.sinceDone--; return; } // short pause at episode end

    this.env.observe(this.obs, 0);
    this.policy.forward(this.obs, 1, this.acts);
    let action: number;
    if (this.greedy) action = argmaxRow(this.acts.logits, 0, N_ACTIONS);
    else {
      softmaxRow(this.acts.logits, 0, N_ACTIONS, this.probs, 0);
      action = sampleCategorical(this.probs, 0, N_ACTIONS, this.rng());
    }
    const res = this.env.step(action);
    this.bot.setTarget(this.env.x, this.env.z, this.env.yaw);

    if (res.done) {
      this.episodes++;
      if (res.success) this.successes++;
      this.onEpisodeEnd?.(res.success);
      // new goal from current position; keep walking from where we are
      this.env.steps = 0;
      this.env.sampleGoal();
      this.env.prevDist = this.env.goalDist();
      this.syncGoal();
      this.sinceDone = res.success ? 5 : 0;
    }
  }
}
