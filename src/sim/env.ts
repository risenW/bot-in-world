// Goal-conditioned navigation environment over a NavGrid.
//
// Everything the policy sees is egocentric (lidar in body frame, goal bearing
// relative to heading, own speed). No absolute coordinates, no world identity —
// that is what lets one policy navigate a world it has never seen.

import { NavGrid, castRay, isWalkable, cellCenter } from './navgrid';
import { Rng } from './rng';

export const N_LIDAR = 16;          // rays, full 360°, body frame
export const LIDAR_RANGE = 6.0;     // meters
export const N_ACTIONS = 6;         // idle, fwd, fwd+L, fwd+R, turn L, turn R
export const OBS_SIZE = N_LIDAR + 5 + N_ACTIONS; // 27

export const DT = 0.1;              // seconds per env step
const MOVE_SPEED = 1.7;             // m/s target
const ACCEL = 8.0;                  // approach rate
const TURN_RATE = 2.6;              // rad/s
const GOAL_RADIUS = 0.55;           // success distance
const MAX_GOAL_OBS = 12.0;          // distance normalization cap

export const MAX_EPISODE_STEPS = 500;

// Reward shaping
const R_PROGRESS = 1.0;             // per meter of progress toward goal
const R_STEP = -0.01;
const R_COLLIDE = -0.05;
const R_SUCCESS = 5.0;

export interface StepResult {
  reward: number;
  done: boolean;       // episode ended (success or timeout)
  success: boolean;
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

  constructor(grid: NavGrid, rng: Rng) {
    this.grid = grid;
    this.rng = rng;
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
    this.prevDist = this.goalDist();
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
    this.prevDist = this.goalDist();
    this.steps = 0;
    return true;
  }

  goalDist(): number {
    return Math.hypot(this.goalX - this.x, this.goalZ - this.z);
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
    this.x = nx; this.z = nz;

    // reward
    const dist = this.goalDist();
    let reward = R_STEP + R_PROGRESS * (this.prevDist - dist);
    if (this.collided) reward += R_COLLIDE;
    this.prevDist = dist;

    const success = dist < GOAL_RADIUS;
    if (success) reward += R_SUCCESS;
    const done = success || this.steps >= MAX_EPISODE_STEPS;
    return { reward, done, success };
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
  }
}
