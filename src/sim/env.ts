// Goal-conditioned environment over a NavGrid with two tasks:
//   'nav'   — walk to the goal point
//   'fetch' — collect every target-colored ball (one at a time) and deliver
//             each to the goal point, fast, without hitting obstacles
//
// Everything the policy sees is egocentric (lidar in body frame, goal/ball
// bearings relative to heading, own speed). No absolute coordinates, no world
// identity — that is what lets one policy navigate a world it has never seen.

import { NavGrid, castRay, isWalkable, cellCenter, groundHeight } from './navgrid';
import { Rng } from './rng';

export const N_LIDAR = 16;          // rays, full 360°, body frame
export const LIDAR_RANGE = 6.0;     // meters
export const N_ACTIONS = 6;         // idle, fwd, fwd+L, fwd+R, turn L, turn R
// 16 lidar + goal(3) + speed + collision + lastAction(6) + carrying + ball(3) + remaining
export const OBS_SIZE = N_LIDAR + 3 + 1 + 1 + N_ACTIONS + 1 + 3 + 1; // 32

export const DT = 0.1;              // seconds per env step
const MOVE_SPEED = 1.7;             // m/s target
const ACCEL = 8.0;                  // approach rate
const TURN_RATE = 2.6;              // rad/s
const GOAL_RADIUS = 0.55;           // deposit / nav-success distance
const PICKUP_RADIUS = 0.45;         // touch a ball to pick it up
const MAX_GOAL_OBS = 12.0;          // distance normalization cap

export const MAX_EPISODE_STEPS = 500;        // nav
export const FETCH_STEPS_PER_BALL = 350;     // fetch timeout = base + per-ball
const FETCH_STEPS_BASE = 300;

// Reward shaping
const R_PROGRESS = 1.0;             // per meter of progress toward current objective
const R_STEP = -0.01;
const R_COLLIDE = -0.05;
const R_SUCCESS = 5.0;              // nav goal reached / all balls delivered
const R_PICKUP = 2.0;
const R_DEPOSIT = 3.0;

export type TaskMode = 'nav' | 'fetch';

export interface TaskConfig {
  mode: TaskMode;
  numBalls: number;        // target-colored balls to collect (fetch)
  numDistractors: number;  // other-colored balls — visual only, never sensed
}

export const DEFAULT_TASK: TaskConfig = { mode: 'nav', numBalls: 4, numDistractors: 3 };

export interface Ball { x: number; z: number; active: boolean }

export interface StepResult {
  reward: number;
  done: boolean;       // episode ended (success or timeout)
  success: boolean;
  pickedUp?: boolean;  // this step picked a ball up
  deposited?: boolean; // this step delivered a ball
}

export class NavEnv {
  grid: NavGrid;
  rng: Rng;

  x = 0; z = 0; yaw = 0; speed = 0;
  goalX = 0; goalZ = 0;
  steps = 0;
  prevDist = 0;
  lastAction = 0;
  collided = false;

  // curriculum: max goal distance in meters, controlled by the trainer
  goalDistMin = 1.5;
  goalDistMax = 4.0;

  // task state
  task: TaskConfig;
  activeBalls: number;     // curriculum-controlled ball count (<= task.numBalls)
  balls: Ball[] = [];
  distractors: Ball[] = [];
  carrying = false;
  delivered = 0;
  climbed = false;         // this step the bot hopped up a climbable surface

  constructor(grid: NavGrid, rng: Rng, task: TaskConfig = DEFAULT_TASK) {
    this.grid = grid;
    this.rng = rng;
    this.task = { ...task };
    this.activeBalls = task.mode === 'fetch' ? task.numBalls : 0;
    this.reset();
  }

  reset(keepPosition = false): void {
    const g = this.grid;
    if (!keepPosition || !isWalkable(g, this.x, this.z)) {
      const ci = g.spawn[(this.rng() * g.spawn.length) | 0];
      const [sx, sz] = cellCenter(g, ci);
      this.x = sx; this.z = sz;
      this.yaw = this.rng() * Math.PI * 2;
      this.speed = 0;
    }
    this.sampleGoal();
    this.steps = 0;
    this.lastAction = 0;
    this.collided = false;
    this.carrying = false;
    this.delivered = 0;
    if (this.task.mode === 'fetch') {
      this.balls = this.sampleBalls(this.activeBalls, 1.2);
      this.distractors = this.sampleBalls(this.task.numDistractors, 0.8);
    } else {
      this.balls = [];
      this.distractors = [];
    }
    this.prevDist = this.objectiveDist();
  }

  // scatter balls on walkable cells, away from the bot and each other
  private sampleBalls(n: number, minBotDist: number): Ball[] {
    const g = this.grid;
    const out: Ball[] = [];
    for (let b = 0; b < n; b++) {
      let bx = this.x, bz = this.z;
      for (let attempt = 0; attempt < 40; attempt++) {
        const ci = g.spawn[(this.rng() * g.spawn.length) | 0];
        const [cx, cz] = cellCenter(g, ci);
        const slack = 1 + attempt * 0.1;
        if (Math.hypot(cx - this.x, cz - this.z) < minBotDist / slack) continue;
        if (Math.hypot(cx - this.goalX, cz - this.goalZ) < GOAL_RADIUS * 2) continue;
        if (out.some((o) => Math.hypot(o.x - cx, o.z - cz) < 0.8 / slack)) continue;
        bx = cx; bz = cz;
        break;
      }
      out.push({ x: bx, z: bz, active: true });
    }
    return out;
  }

  sampleGoal(): void {
    const g = this.grid;
    // Rejection-sample a goal in the curriculum distance band; relax if unlucky.
    for (let attempt = 0; attempt < 60; attempt++) {
      const ci = g.spawn[(this.rng() * g.spawn.length) | 0];
      const [gx, gz] = cellCenter(g, ci);
      const d = Math.hypot(gx - this.x, gz - this.z);
      const slack = 1 + attempt * 0.15;
      if (d >= this.goalDistMin / slack && d <= this.goalDistMax * slack) {
        this.goalX = gx; this.goalZ = gz;
        return;
      }
    }
    const ci = g.spawn[(this.rng() * g.spawn.length) | 0];
    [this.goalX, this.goalZ] = cellCenter(g, ci);
  }

  setGoal(x: number, z: number): boolean {
    if (!isWalkable(this.grid, x, z)) return false;
    this.goalX = x; this.goalZ = z;
    this.prevDist = this.objectiveDist();
    this.steps = 0;
    return true;
  }

  goalDist(): number {
    return Math.hypot(this.goalX - this.x, this.goalZ - this.z);
  }

  nearestBall(): Ball | null {
    let best: Ball | null = null, bd = Infinity;
    for (const b of this.balls) {
      if (!b.active) continue;
      const d = Math.hypot(b.x - this.x, b.z - this.z);
      if (d < bd) { bd = d; best = b; }
    }
    return best;
  }

  // distance to the current objective: nearest ball when hunting, goal otherwise
  objectiveDist(): number {
    if (this.task.mode === 'fetch' && !this.carrying) {
      const b = this.nearestBall();
      if (b) return Math.hypot(b.x - this.x, b.z - this.z);
    }
    return this.goalDist();
  }

  maxSteps(): number {
    return this.task.mode === 'fetch'
      ? FETCH_STEPS_BASE + FETCH_STEPS_PER_BALL * Math.max(1, this.activeBalls)
      : MAX_EPISODE_STEPS;
  }

  step(action: number): StepResult {
    const g = this.grid;
    this.steps++;
    this.lastAction = action;
    this.collided = false;

    // turning
    if (action === 2 || action === 4) this.yaw += TURN_RATE * DT;
    if (action === 3 || action === 5) this.yaw -= TURN_RATE * DT;
    if (this.yaw > Math.PI) this.yaw -= Math.PI * 2;
    if (this.yaw < -Math.PI) this.yaw += Math.PI * 2;

    // forward speed (actions 1,2,3 drive)
    const target = action === 1 || action === 2 || action === 3 ? MOVE_SPEED : 0;
    this.speed += (target - this.speed) * Math.min(1, ACCEL * DT);

    // attempted move with axis sliding
    const dx = Math.sin(this.yaw) * this.speed * DT;
    const dz = Math.cos(this.yaw) * this.speed * DT;
    let nx = this.x + dx, nz = this.z + dz;
    if (!isWalkable(g, nx, nz)) {
      this.collided = this.speed > 0.15;
      if (isWalkable(g, this.x + dx, this.z)) { nx = this.x + dx; nz = this.z; }
      else if (isWalkable(g, this.x, this.z + dz)) { nx = this.x; nz = this.z + dz; }
      else { nx = this.x; nz = this.z; this.speed = 0; }
    }
    // climbing: stepping up costs momentum (the bot jumps like a humanoid)
    this.climbed = false;
    const gOld = groundHeight(g, this.x, this.z);
    const gNew = groundHeight(g, nx, nz);
    if (!Number.isNaN(gOld) && !Number.isNaN(gNew) && gNew - gOld > 0.26) {
      this.climbed = true;
      this.speed *= 0.35;
    }
    this.x = nx; this.z = nz;

    // progress shaping toward the current objective (ball when hunting,
    // goal when carrying or in nav mode)
    let dist = this.objectiveDist();
    let reward = R_STEP + R_PROGRESS * (this.prevDist - dist);
    if (this.collided) reward += R_COLLIDE;

    let success = false, pickedUp = false, deposited = false;
    if (this.task.mode === 'fetch') {
      if (!this.carrying) {
        const b = this.nearestBall();
        if (b && Math.hypot(b.x - this.x, b.z - this.z) < PICKUP_RADIUS) {
          b.active = false;
          this.carrying = true;
          pickedUp = true;
          reward += R_PICKUP;
          dist = this.objectiveDist(); // objective switches to the goal
        }
      } else if (this.goalDist() < GOAL_RADIUS) {
        this.carrying = false;
        this.delivered++;
        deposited = true;
        reward += R_DEPOSIT;
        if (this.delivered >= this.balls.length) {
          success = true;
          reward += R_SUCCESS;
        } else {
          dist = this.objectiveDist(); // objective switches to the next ball
        }
      }
    } else {
      success = dist < GOAL_RADIUS;
      if (success) reward += R_SUCCESS;
    }
    this.prevDist = dist;

    const done = success || this.steps >= this.maxSteps();
    return { reward, done, success, pickedUp, deposited };
  }

  // Write observation into out[offset..offset+OBS_SIZE)
  observe(out: Float32Array, offset: number): void {
    const g = this.grid;
    for (let i = 0; i < N_LIDAR; i++) {
      const ang = this.yaw + (i / N_LIDAR) * Math.PI * 2;
      const d = castRay(g, this.x, this.z, Math.sin(ang), Math.cos(ang), LIDAR_RANGE);
      out[offset + i] = d / LIDAR_RANGE;
    }
    const gdx = this.goalX - this.x, gdz = this.goalZ - this.z;
    const dist = Math.hypot(gdx, gdz);
    const bearing = Math.atan2(gdx, gdz) - this.yaw; // 0 = straight ahead
    let o = offset + N_LIDAR;
    out[o++] = Math.min(dist, MAX_GOAL_OBS) / MAX_GOAL_OBS;
    out[o++] = Math.sin(bearing);
    out[o++] = Math.cos(bearing);
    out[o++] = this.speed / MOVE_SPEED;
    out[o++] = this.collided ? 1 : 0;
    for (let a = 0; a < N_ACTIONS; a++) out[o + a] = a === this.lastAction ? 1 : 0;
    o += N_ACTIONS;

    // fetch-task channels (zeros in nav mode, except ballDist parked at max)
    const ball = this.task.mode === 'fetch' && !this.carrying ? this.nearestBall() : null;
    out[o++] = this.carrying ? 1 : 0;
    if (ball) {
      const bdx = ball.x - this.x, bdz = ball.z - this.z;
      const bDist = Math.hypot(bdx, bdz);
      const bBearing = Math.atan2(bdx, bdz) - this.yaw;
      out[o++] = Math.min(bDist, MAX_GOAL_OBS) / MAX_GOAL_OBS;
      out[o++] = Math.sin(bBearing);
      out[o++] = Math.cos(bBearing);
    } else {
      out[o++] = 1;
      out[o++] = 0;
      out[o++] = 0;
    }
    out[o++] = this.task.mode === 'fetch' && this.balls.length > 0
      ? (this.balls.length - this.delivered - (this.carrying ? 1 : 0)) / this.balls.length
      : 0;
  }
}
