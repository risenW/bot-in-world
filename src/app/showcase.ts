// Showcase: a live NavEnv on the main thread driven by the latest policy
// weights — this is the bot you watch while (or after) training.

import * as pc from 'playcanvas';
import { NavEnv, OBS_SIZE, N_ACTIONS, DT, TaskConfig, DEFAULT_TASK } from '../sim/env';
import { NavGrid, groundHeight, isWalkable, castRay, cellCenter } from '../sim/navgrid';
import { Policy, makeActivations, Activations, softmaxRow, sampleCategorical, argmaxRow } from '../sim/ppo';
import { mulberry32, Rng } from '../sim/rng';
import { HumanoidBot, GoalMarker } from './bot';

// fetch-task ball visuals: the bot only "senses" TARGET_COLOR balls —
// distractors are scenery it must ignore
export const TARGET_COLOR = new pc.Color(0.31, 0.76, 0.97);   // puffer cyan
const DISTRACTOR_COLORS = [
  new pc.Color(0.94, 0.33, 0.31), new pc.Color(1.0, 0.93, 0.35), new pc.Color(0.67, 0.28, 0.74),
];
const BALL_R = 0.14;

function makeBallEntity(app: pc.Application, color: pc.Color): pc.Entity {
  const m = new pc.StandardMaterial();
  m.diffuse = color;
  m.emissive = new pc.Color(color.r * 0.45, color.g * 0.45, color.b * 0.45);
  m.gloss = 0.7;
  m.update();
  const e = new pc.Entity('task-ball');
  e.addComponent('render', { type: 'sphere', material: m });
  e.setLocalScale(BALL_R * 2, BALL_R * 2, BALL_R * 2);
  app.root.addChild(e);
  return e;
}

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
  onTaskEvent: ((event: 'pickup' | 'deposit', remaining: number) => void) | null = null;

  task: TaskConfig = { ...DEFAULT_TASK };
  private app: pc.Application;
  private ballEntities: pc.Entity[] = [];
  private carryBall: pc.Entity;

  constructor(app: pc.Application, grid: NavGrid) {
    this.app = app;
    this.grid = grid;
    this.rng = mulberry32(Math.floor(Math.random() * 1e9));
    this.env = this.makeEnv();
    this.policy = new Policy(OBS_SIZE, 128, N_ACTIONS, this.rng);
    this.acts = makeActivations(this.policy, 1);
    this.bot = new HumanoidBot(app);
    this.goal = new GoalMarker(app);
    this.carryBall = makeBallEntity(app, TARGET_COLOR);
    this.carryBall.enabled = false;
    this.respawn(true);
  }

  private makeEnv(): NavEnv {
    const env = new NavEnv(this.grid, this.rng, this.task);
    env.goalDistMin = 2;
    env.goalDistMax = 9;
    return env;
  }

  setGrid(grid: NavGrid): void {
    this.grid = grid;
    this.env = this.makeEnv();
    this.episodes = 0; this.successes = 0;
    this.respawn(true);
  }

  setTask(task: TaskConfig): void {
    this.task = { ...task };
    const x = this.env.x, z = this.env.z, yaw = this.env.yaw;
    this.env = this.makeEnv();
    this.env.x = x; this.env.z = z; this.env.yaw = yaw;
    this.env.reset(true);
    this.episodes = 0; this.successes = 0;
    this.bot.snapTo(this.env.x, this.env.z, this.env.yaw);
    this.syncGoal();
    this.syncBalls();
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

  newGoal(): void {
    this.env.steps = 0;
    this.env.sampleGoal();
    this.env.prevDist = this.env.goalDist();
    this.syncGoal();
  }

  clickGoal(x: number, z: number): boolean {
    if (!isWalkable(this.grid, x, z)) return false;
    this.env.setGoal(x, z);
    this.syncGoal();
    return true;
  }

  // centered=true spawns near the world origin (best splat quality) — used on
  // world load so the first thing people see isn't a degraded edge.
  respawn(centered = false): void {
    if (centered) {
      const ci = this.nearestCenterCell();
      const [cx, cz] = cellCenter(this.grid, ci);
      this.env.x = cx; this.env.z = cz;
      this.env.yaw = this.rng() * Math.PI * 2;
      this.env.speed = 0;
      // keep the first goal central too, so the bot doesn't immediately walk
      // out toward a degraded edge
      const prevMax = this.env.goalDistMax;
      this.env.goalDistMax = Math.min(prevMax, 6);
      this.env.reset(true); // keep the centered position, fresh goal/balls
      this.env.goalDistMax = prevMax;
    } else {
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
    }
    this.bot.snapTo(this.env.x, this.env.z, this.env.yaw);
    this.syncGoal();
    this.syncBalls();
  }

  // walkable spawn cell closest to the world XZ origin (the splat capture point)
  private nearestCenterCell(): number {
    const g = this.grid;
    let best = g.spawn[0], bd = Infinity;
    for (const ci of g.spawn) {
      const [x, z] = cellCenter(g, ci);
      const d = x * x + z * z;
      if (d < bd) { bd = d; best = ci; }
    }
    return best;
  }

  private syncGoal(): void {
    this.goal.setPosition(this.env.goalX, groundHeight(this.grid, this.env.goalX, this.env.goalZ), this.env.goalZ);
  }

  // (re)build ball entities to match the env's current episode
  private syncBalls(): void {
    for (const e of this.ballEntities) e.destroy();
    this.ballEntities = [];
    const all = [
      ...this.env.balls.map((b) => ({ b, color: TARGET_COLOR })),
      ...this.env.distractors.map((b, i) => ({ b, color: DISTRACTOR_COLORS[i % DISTRACTOR_COLORS.length] })),
    ];
    for (const { b, color } of all) {
      const e = makeBallEntity(this.app, color);
      const gy = groundHeight(this.grid, b.x, b.z);
      e.setPosition(b.x, (Number.isNaN(gy) ? 0 : gy) + BALL_R, b.z);
      this.ballEntities.push(e);
    }
    this.refreshBallVisibility();
  }

  private refreshBallVisibility(): void {
    for (let i = 0; i < this.env.balls.length; i++) {
      if (this.ballEntities[i]) this.ballEntities[i].enabled = this.env.balls[i].active;
    }
    this.carryBall.enabled = this.env.carrying;
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
    if (this.carryBall.enabled) {
      const p = this.bot.position;
      this.carryBall.setPosition(p.x, p.y + 1.75 + Math.sin(performance.now() / 300) * 0.04, p.z);
    }
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
    if (this.env.climbed) this.bot.triggerJump();

    if (res.pickedUp || res.deposited) {
      this.refreshBallVisibility();
      const remaining = this.env.balls.filter((b) => b.active).length + (this.env.carrying ? 1 : 0);
      this.onTaskEvent?.(res.pickedUp ? 'pickup' : 'deposit', remaining);
    }

    if (res.done) {
      this.episodes++;
      if (res.success) this.successes++;
      this.onEpisodeEnd?.(res.success);
      // fresh episode from where we are: new goal (and new balls in fetch mode)
      this.env.reset(true);
      this.syncGoal();
      this.syncBalls();
      this.sinceDone = res.success ? 8 : 0;
    }
  }
}
