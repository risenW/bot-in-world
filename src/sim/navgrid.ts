// NavGrid: bakes a collision mesh into a 2D walkability + ground-height field.
// This is the world abstraction the RL environment runs on. Observations built
// from it are purely egocentric, so a policy trained on one grid transfers to
// any other world baked the same way.

import type { ParsedMesh } from './ply';

export interface NavGrid {
  cell: number;            // cell size in meters
  minX: number;
  minZ: number;
  w: number;               // cells along X
  h: number;               // cells along Z
  walkable: Uint8Array;    // 1 = bot center may occupy (eroded by bot radius)
  ground: Float32Array;    // ground Y per cell (NaN where none)
  spawn: Uint32Array;      // cell indices of the largest connected walkable region
  meshMinY: number;
}

const CELL = 0.15;          // meters per cell
const BOT_RADIUS = 0.26;    // erosion radius
const CLEAR_LOW = 0.3;      // obstacle band above ground...
const CLEAR_HIGH = 1.65;    // ...blocks the cell (bot is ~1.5m tall)
const MAX_STEP = 0.22;      // max ground height delta between neighbor cells
const FLOOR_NY = 0.55;      // |triangle normal Y| to count as floor

export function bakeNavGrid(mesh: ParsedMesh): NavGrid {
  const { positions: P, indices: I, aabb } = mesh;
  const minX = aabb.min[0] - CELL, minZ = aabb.min[2] - CELL;
  const w = Math.max(4, Math.ceil((aabb.max[0] - minX + CELL) / CELL));
  const h = Math.max(4, Math.ceil((aabb.max[2] - minZ + CELL) / CELL));
  const nCells = w * h;
  const nTris = I.length / 3;

  // --- bucket triangles by XZ AABB (counting sort into flat arrays) ---
  const counts = new Uint32Array(nCells + 1);
  const triCellSpan = new Int32Array(nTris * 4); // cx0, cz0, cx1, cz1
  for (let t = 0; t < nTris; t++) {
    const a = I[t * 3] * 3, b = I[t * 3 + 1] * 3, c = I[t * 3 + 2] * 3;
    const x0 = Math.min(P[a], P[b], P[c]), x1 = Math.max(P[a], P[b], P[c]);
    const z0 = Math.min(P[a + 2], P[b + 2], P[c + 2]), z1 = Math.max(P[a + 2], P[b + 2], P[c + 2]);
    const cx0 = Math.max(0, Math.floor((x0 - minX) / CELL)), cx1 = Math.min(w - 1, Math.floor((x1 - minX) / CELL));
    const cz0 = Math.max(0, Math.floor((z0 - minZ) / CELL)), cz1 = Math.min(h - 1, Math.floor((z1 - minZ) / CELL));
    triCellSpan[t * 4] = cx0; triCellSpan[t * 4 + 1] = cz0; triCellSpan[t * 4 + 2] = cx1; triCellSpan[t * 4 + 3] = cz1;
    for (let cz = cz0; cz <= cz1; cz++) for (let cx = cx0; cx <= cx1; cx++) counts[cz * w + cx + 1]++;
  }
  for (let i = 1; i <= nCells; i++) counts[i] += counts[i - 1];
  const bucketTris = new Uint32Array(counts[nCells]);
  const cursor = counts.slice(0, nCells);
  for (let t = 0; t < nTris; t++) {
    const cx0 = triCellSpan[t * 4], cz0 = triCellSpan[t * 4 + 1], cx1 = triCellSpan[t * 4 + 2], cz1 = triCellSpan[t * 4 + 3];
    for (let cz = cz0; cz <= cz1; cz++) for (let cx = cx0; cx <= cx1; cx++) bucketTris[cursor[cz * w + cx]++] = t;
  }

  // --- per-triangle plane data + floor flag ---
  const triFloor = new Uint8Array(nTris);
  for (let t = 0; t < nTris; t++) {
    const a = I[t * 3] * 3, b = I[t * 3 + 1] * 3, c = I[t * 3 + 2] * 3;
    const abx = P[b] - P[a], aby = P[b + 1] - P[a + 1], abz = P[b + 2] - P[a + 2];
    const acx = P[c] - P[a], acy = P[c + 1] - P[a + 1], acz = P[c + 2] - P[a + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const len = Math.hypot(nx, ny, nz) || 1;
    if (Math.abs(ny / len) >= FLOOR_NY) triFloor[t] = 1;
  }

  // --- sample each cell center: vertical line vs triangles ---
  const ground = new Float32Array(nCells).fill(NaN);
  const blocked = new Uint8Array(nCells);
  const hitsY: number[] = [];
  const hitsFloor: number[] = [];
  for (let cz = 0; cz < h; cz++) {
    for (let cx = 0; cx < w; cx++) {
      const ci = cz * w + cx;
      const px = minX + (cx + 0.5) * CELL;
      const pz = minZ + (cz + 0.5) * CELL;
      hitsY.length = 0; hitsFloor.length = 0;
      for (let k = counts[ci]; k < counts[ci + 1]; k++) {
        const t = bucketTris[k];
        const ia = I[t * 3] * 3, ib = I[t * 3 + 1] * 3, ic = I[t * 3 + 2] * 3;
        const y = intersectVertical(px, pz, P, ia, ib, ic);
        if (!Number.isNaN(y)) { hitsY.push(y); hitsFloor.push(triFloor[t]); }
      }
      if (hitsY.length === 0) continue;
      // ground = lowest floor-ish hit
      let g = NaN;
      for (let i = 0; i < hitsY.length; i++) {
        if (hitsFloor[i] && (Number.isNaN(g) || hitsY[i] < g)) g = hitsY[i];
      }
      if (Number.isNaN(g)) { blocked[ci] = 1; continue; }
      ground[ci] = g;
      for (let i = 0; i < hitsY.length; i++) {
        const dy = hitsY[i] - g;
        if (dy > CLEAR_LOW && dy < CLEAR_HIGH) { blocked[ci] = 1; break; }
      }
    }
  }

  // --- walkability: ground + not blocked + step constraint ---
  const walk = new Uint8Array(nCells);
  for (let ci = 0; ci < nCells; ci++) walk[ci] = blocked[ci] === 0 && !Number.isNaN(ground[ci]) ? 1 : 0;
  for (let cz = 0; cz < h; cz++) {
    for (let cx = 0; cx < w; cx++) {
      const ci = cz * w + cx;
      if (!walk[ci]) continue;
      const g = ground[ci];
      let bad = false;
      for (let dz = -1; dz <= 1 && !bad; dz++) {
        for (let dx = -1; dx <= 1 && !bad; dx++) {
          const nx = cx + dx, nz = cz + dz;
          if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
          const ni = nz * w + nx;
          if (walk[ni] && Math.abs(ground[ni] - g) > MAX_STEP) bad = true;
        }
      }
      if (bad) walk[ci] = 0;
    }
  }

  // --- erode by bot radius (dilate blocked) ---
  const erodeCells = Math.ceil(BOT_RADIUS / CELL);
  const eroded = new Uint8Array(walk);
  for (let cz = 0; cz < h; cz++) {
    for (let cx = 0; cx < w; cx++) {
      if (walk[cz * w + cx]) continue;
      for (let dz = -erodeCells; dz <= erodeCells; dz++) {
        for (let dx = -erodeCells; dx <= erodeCells; dx++) {
          if (dx * dx + dz * dz > erodeCells * erodeCells) continue;
          const nx = cx + dx, nz = cz + dz;
          if (nx >= 0 && nz >= 0 && nx < w && nz < h) eroded[nz * w + nx] = 0;
        }
      }
    }
  }

  // --- largest connected component (4-neighbor BFS) ---
  const comp = new Int32Array(nCells).fill(-1);
  let bestComp = -1, bestSize = 0, nComp = 0;
  const queue = new Int32Array(nCells);
  for (let start = 0; start < nCells; start++) {
    if (!eroded[start] || comp[start] !== -1) continue;
    let head = 0, tail = 0, size = 0;
    queue[tail++] = start; comp[start] = nComp;
    while (head < tail) {
      const ci = queue[head++]; size++;
      const cx = ci % w, cz = (ci / w) | 0;
      if (cx > 0 && eroded[ci - 1] && comp[ci - 1] === -1) { comp[ci - 1] = nComp; queue[tail++] = ci - 1; }
      if (cx < w - 1 && eroded[ci + 1] && comp[ci + 1] === -1) { comp[ci + 1] = nComp; queue[tail++] = ci + 1; }
      if (cz > 0 && eroded[ci - w] && comp[ci - w] === -1) { comp[ci - w] = nComp; queue[tail++] = ci - w; }
      if (cz < h - 1 && eroded[ci + w] && comp[ci + w] === -1) { comp[ci + w] = nComp; queue[tail++] = ci + w; }
    }
    if (size > bestSize) { bestSize = size; bestComp = nComp; }
    nComp++;
  }
  const spawn: number[] = [];
  const final = new Uint8Array(nCells);
  for (let ci = 0; ci < nCells; ci++) {
    if (eroded[ci] && comp[ci] === bestComp) { final[ci] = 1; spawn.push(ci); }
  }

  return {
    cell: CELL, minX, minZ, w, h,
    walkable: final, ground, spawn: new Uint32Array(spawn),
    meshMinY: aabb.min[1],
  };
}

function intersectVertical(px: number, pz: number, P: Float32Array, ia: number, ib: number, ic: number): number {
  // 2D barycentric in XZ; returns interpolated Y or NaN
  const ax = P[ia], az = P[ia + 2], bx = P[ib], bz = P[ib + 2], cx = P[ic], cz = P[ic + 2];
  const v0x = bx - ax, v0z = bz - az;
  const v1x = cx - ax, v1z = cz - az;
  const v2x = px - ax, v2z = pz - az;
  const den = v0x * v1z - v1x * v0z;
  if (Math.abs(den) < 1e-12) return NaN;
  const u = (v2x * v1z - v1x * v2z) / den;
  const v = (v0x * v2z - v2x * v0z) / den;
  if (u < -1e-6 || v < -1e-6 || u + v > 1 + 1e-6) return NaN;
  return P[ia + 1] + u * (P[ib + 1] - P[ia + 1]) + v * (P[ic + 1] - P[ia + 1]);
}

// ---------------- queries ----------------

export function cellIndex(g: NavGrid, x: number, z: number): number {
  const cx = Math.floor((x - g.minX) / g.cell);
  const cz = Math.floor((z - g.minZ) / g.cell);
  if (cx < 0 || cz < 0 || cx >= g.w || cz >= g.h) return -1;
  return cz * g.w + cx;
}

export function isWalkable(g: NavGrid, x: number, z: number): boolean {
  const ci = cellIndex(g, x, z);
  return ci >= 0 && g.walkable[ci] === 1;
}

export function groundHeight(g: NavGrid, x: number, z: number): number {
  const ci = cellIndex(g, x, z);
  if (ci < 0) return NaN;
  const gy = g.ground[ci];
  return Number.isNaN(gy) ? g.meshMinY : gy;
}

// DDA ray over the walkable field; returns distance to first non-walkable cell.
export function castRay(g: NavGrid, x: number, z: number, dx: number, dz: number, maxDist: number): number {
  let cx = Math.floor((x - g.minX) / g.cell);
  let cz = Math.floor((z - g.minZ) / g.cell);
  if (cx < 0 || cz < 0 || cx >= g.w || cz >= g.h) return 0;
  const stepX = dx > 0 ? 1 : -1, stepZ = dz > 0 ? 1 : -1;
  const tDeltaX = dx !== 0 ? Math.abs(g.cell / dx) : Infinity;
  const tDeltaZ = dz !== 0 ? Math.abs(g.cell / dz) : Infinity;
  const bx = g.minX + cx * g.cell, bz = g.minZ + cz * g.cell;
  let tMaxX = dx !== 0 ? (dx > 0 ? (bx + g.cell - x) / dx : (bx - x) / dx) : Infinity;
  let tMaxZ = dz !== 0 ? (dz > 0 ? (bz + g.cell - z) / dz : (bz - z) / dz) : Infinity;
  let t = 0;
  for (let iter = 0; iter < 4096; iter++) {
    if (g.walkable[cz * g.w + cx] === 0) return Math.min(t, maxDist);
    if (tMaxX < tMaxZ) { t = tMaxX; tMaxX += tDeltaX; cx += stepX; }
    else { t = tMaxZ; tMaxZ += tDeltaZ; cz += stepZ; }
    if (t >= maxDist) return maxDist;
    if (cx < 0 || cz < 0 || cx >= g.w || cz >= g.h) return Math.min(t, maxDist);
  }
  return maxDist;
}

export function cellCenter(g: NavGrid, ci: number): [number, number] {
  const cx = ci % g.w, cz = (ci / g.w) | 0;
  return [g.minX + (cx + 0.5) * g.cell, g.minZ + (cz + 0.5) * g.cell];
}

// ---------------- (de)serialization — structured-clone friendly ----------------

export interface NavGridTransfer {
  cell: number; minX: number; minZ: number; w: number; h: number; meshMinY: number;
  walkable: Uint8Array; ground: Float32Array; spawn: Uint32Array;
}

export function toTransfer(g: NavGrid): NavGridTransfer { return { ...g }; }
export function fromTransfer(t: NavGridTransfer): NavGrid { return { ...t }; }
