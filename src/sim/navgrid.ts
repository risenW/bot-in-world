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
  climbHeight: number;     // max ground step the bot may climb (0 = stairs only)
}

export interface BakeOptions {
  climbHeight?: number;    // promote low flat obstacles (crates, platforms) to walkable
}

const CELL = 0.15;          // meters per cell
const BOT_RADIUS = 0.26;    // erosion radius
const CLEAR_LOW = 0.3;      // obstacle band above ground...
const CLEAR_HIGH = 1.65;    // ...blocks the cell (bot is ~1.5m tall)
const MAX_STEP = 0.22;      // max ground height delta between neighbor cells
const FLOOR_NY = 0.55;      // |triangle normal Y| to count as floor
const SUPPORT_BAND = 2.2;   // splat points within [-0.3, +2.2] of a floor candidate support it
const MIN_SUPPORT = 30;     // supporting splat points (3x3 neighborhood) for a valid floor

// `splatPoints` (xyz triplets of the gaussian splat centers) drives the
// spawn filter: the reconstructed mesh extends far beyond the actual world
// (flat void planes outside the walls), but gaussians only exist where the
// world is real. Without it, spawning falls back to the full walkable region.
export function bakeNavGrid(mesh: ParsedMesh, splatPoints?: Float32Array, opts: BakeOptions = {}): NavGrid {
  const climbHeight = opts.climbHeight ?? 0;
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

  // --- bucket splat points per cell (when provided) ---
  // The reconstructed mesh is a closed "balloon" that extends far beyond the
  // visible world: phantom floor planes outside the walls, and surfaces below
  // the real floor. Gaussians only exist where the world is real, so a floor
  // candidate is only valid if splat points sit just above it.
  let sCounts: Uint32Array | null = null;
  let sYs: Float32Array | null = null;
  if (splatPoints && splatPoints.length >= 3) {
    sCounts = new Uint32Array(nCells + 1);
    const cellOf = (i: number) => {
      const cx = Math.floor((splatPoints[i] - minX) / CELL);
      const cz = Math.floor((splatPoints[i + 2] - minZ) / CELL);
      return cx < 0 || cz < 0 || cx >= w || cz >= h ? -1 : cz * w + cx;
    };
    for (let i = 0; i < splatPoints.length; i += 3) {
      const ci = cellOf(i);
      if (ci >= 0) sCounts[ci + 1]++;
    }
    for (let i = 1; i <= nCells; i++) sCounts[i] += sCounts[i - 1];
    sYs = new Float32Array(sCounts[nCells]);
    const sCursor = sCounts.slice(0, nCells);
    for (let i = 0; i < splatPoints.length; i += 3) {
      const ci = cellOf(i);
      if (ci >= 0) sYs[sCursor[ci]++] = splatPoints[i + 1];
    }
  }
  // supporting splat points in the 3x3 neighborhood within [y+lo, y+hi]
  const support = (cx: number, cz: number, y: number): number => {
    if (!sCounts || !sYs) return MIN_SUPPORT; // no splat data -> everything passes
    let n = 0;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const x = cx + dx, z = cz + dz;
        if (x < 0 || z < 0 || x >= w || z >= h) continue;
        const ci = z * w + x;
        for (let k = sCounts[ci]; k < sCounts[ci + 1]; k++) {
          const dy = sYs[k] - y;
          if (dy > -0.3 && dy < SUPPORT_BAND) n++;
        }
      }
    }
    return n;
  };

  // --- sample each cell center: vertical line vs triangles ---
  const ground = new Float32Array(nCells).fill(NaN);
  const blocked = new Uint8Array(nCells);
  const protect = new Uint8Array(nCells); // climb cells: reduced erosion
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
      // ground = lowest floor-ish hit with splat support
      let g = NaN;
      for (let i = 0; i < hitsY.length; i++) {
        if (!hitsFloor[i]) continue;
        if (!Number.isNaN(g) && hitsY[i] >= g) continue;
        if (support(cx, cz, hitsY[i]) >= MIN_SUPPORT) g = hitsY[i];
      }
      if (Number.isNaN(g)) { blocked[ci] = 1; continue; }
      ground[ci] = g;
      for (let i = 0; i < hitsY.length; i++) {
        const dy = hitsY[i] - g;
        if (dy > CLEAR_LOW && dy < CLEAR_HIGH) { blocked[ci] = 1; break; }
      }

      // climbable promotion: a blocked cell whose blocking surface is itself a
      // low flat top (crate, platform, step) within climbHeight, with standing
      // clearance and splat support, becomes elevated ground.
      if (blocked[ci] && climbHeight > 0) {
        let top = NaN;
        for (let i = 0; i < hitsY.length; i++) {
          const dy = hitsY[i] - g;
          if (!hitsFloor[i] || dy <= CLEAR_LOW || dy > climbHeight) continue;
          if (Number.isNaN(top) || hitsY[i] > top) top = hitsY[i];
        }
        if (!Number.isNaN(top) && support(cx, cz, top) >= MIN_SUPPORT) {
          let clear = true;
          for (let i = 0; i < hitsY.length; i++) {
            const dy = hitsY[i] - top;
            if (dy > CLEAR_LOW && dy < CLEAR_HIGH) { clear = false; break; }
          }
          if (clear) { ground[ci] = top; blocked[ci] = 0; protect[ci] = 1; }
        }
      }
    }
  }

  // --- climb seam bridging ---
  // A crate/platform edge shows up as a thin ring of blocked cells (vertical
  // side surfaces) separating the floor from the climbable top, which would
  // disconnect the top. Bridge: a blocked cell whose opposite neighbors are
  // walkable at two levels within climbHeight becomes walkable at the upper
  // level. Two passes handle 2-cell-thick rims and corners.
  if (climbHeight > 0) {
    const okCell = (ci: number) => blocked[ci] === 0 && !Number.isNaN(ground[ci]);
    for (let iter = 0; iter < 2; iter++) {
      for (let cz = 1; cz < h - 1; cz++) {
        for (let cx = 1; cx < w - 1; cx++) {
          const ci = cz * w + cx;
          if (okCell(ci)) continue;
          for (const [a, b] of [[ci - 1, ci + 1], [ci - w, ci + w], [ci - w - 1, ci + w + 1], [ci - w + 1, ci + w - 1]]) {
            if (!okCell(a) || !okCell(b)) continue;
            const delta = Math.abs(ground[a] - ground[b]);
            if (delta < 0.18 || delta > climbHeight) continue;
            ground[ci] = Math.max(ground[a], ground[b]);
            blocked[ci] = 0;
            protect[ci] = 1;
            break;
          }
        }
      }
    }
  }

  return finalizeGrid(ground, blocked, w, h, minX, minZ, aabb.min[1], climbHeight, protect);
}

// Bake a navgrid from gaussian splat points alone — for worlds loaded from a
// public Spaitial link, where no reconstructed mesh export is available.
// Floor = lowest dense band of points per column; obstacle = points at body
// height above that floor.
const SPLAT_FLOOR_PTS = 20;   // 3x3-neighborhood points in a 0.3m band to call it floor
const SPLAT_BLOCK_PTS = 5;    // own-cell points at body height to call it blocked
export function bakeNavGridFromSplat(points: Float32Array): NavGrid {
  let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity, minY = Infinity;
  for (let i = 0; i < points.length; i += 3) {
    if (points[i] < minX) minX = points[i];
    if (points[i] > maxX) maxX = points[i];
    if (points[i + 2] < minZ) minZ = points[i + 2];
    if (points[i + 2] > maxZ) maxZ = points[i + 2];
    if (points[i + 1] < minY) minY = points[i + 1];
  }
  minX -= CELL; minZ -= CELL;
  const w = Math.max(4, Math.ceil((maxX - minX + CELL) / CELL));
  const h = Math.max(4, Math.ceil((maxZ - minZ + CELL) / CELL));
  const nCells = w * h;

  // bucket point Ys per cell
  const counts = new Uint32Array(nCells + 1);
  const cellOf = (i: number) => {
    const cx = Math.floor((points[i] - minX) / CELL);
    const cz = Math.floor((points[i + 2] - minZ) / CELL);
    return cx < 0 || cz < 0 || cx >= w || cz >= h ? -1 : cz * w + cx;
  };
  for (let i = 0; i < points.length; i += 3) {
    const ci = cellOf(i);
    if (ci >= 0) counts[ci + 1]++;
  }
  for (let i = 1; i <= nCells; i++) counts[i] += counts[i - 1];
  const ys = new Float32Array(counts[nCells]);
  const cursor = counts.slice(0, nCells);
  for (let i = 0; i < points.length; i += 3) {
    const ci = cellOf(i);
    if (ci >= 0) ys[cursor[ci]++] = points[i + 1];
  }

  const ground = new Float32Array(nCells).fill(NaN);
  const blocked = new Uint8Array(nCells);
  const neighborYs: number[] = [];
  for (let cz = 0; cz < h; cz++) {
    for (let cx = 0; cx < w; cx++) {
      const ci = cz * w + cx;
      // gather 3x3 neighborhood Ys
      neighborYs.length = 0;
      for (let dz = -1; dz <= 1; dz++) {
        for (let dx = -1; dx <= 1; dx++) {
          const x = cx + dx, z = cz + dz;
          if (x < 0 || z < 0 || x >= w || z >= h) continue;
          const ni = z * w + x;
          for (let k = counts[ni]; k < counts[ni + 1]; k++) neighborYs.push(ys[k]);
        }
      }
      if (neighborYs.length < SPLAT_FLOOR_PTS) continue;
      neighborYs.sort((a, b) => a - b);
      // floor = lowest y such that >= SPLAT_FLOOR_PTS points lie within +0.3m
      let floor = NaN;
      for (let i = 0; i + SPLAT_FLOOR_PTS - 1 < neighborYs.length; i++) {
        if (neighborYs[i + SPLAT_FLOOR_PTS - 1] - neighborYs[i] <= 0.3) {
          floor = neighborYs[i];
          break;
        }
      }
      if (Number.isNaN(floor)) continue;
      ground[ci] = floor;
      // own-cell content at body height blocks
      let nBlock = 0;
      for (let k = counts[ci]; k < counts[ci + 1]; k++) {
        const dy = ys[k] - floor;
        if (dy > CLEAR_LOW && dy < CLEAR_HIGH) nBlock++;
      }
      if (nBlock >= SPLAT_BLOCK_PTS) blocked[ci] = 1;
    }
  }

  return finalizeGrid(ground, blocked, w, h, minX, minZ, minY, 0);
}

// Shared post-processing: walkability + step constraint + erosion + largest
// connected component -> final grid.
function finalizeGrid(
  ground: Float32Array, blocked: Uint8Array,
  w: number, h: number, minX: number, minZ: number, meshMinY: number,
  climbHeight: number,
  protect?: Uint8Array,
): NavGrid {
  const nCells = w * h;
  const maxStep = Math.max(MAX_STEP, climbHeight);

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
          if (walk[ni] && Math.abs(ground[ni] - g) > maxStep) bad = true;
        }
      }
      if (bad) walk[ci] = 0;
    }
  }

  // --- erode by bot radius (dilate blocked) ---
  // Climb cells (crate tops, seams) only take 1-cell erosion: full bot-radius
  // erosion would erase every narrow climbable top.
  const erodeCells = Math.ceil(BOT_RADIUS / CELL);
  const eroded = new Uint8Array(walk);
  for (let cz = 0; cz < h; cz++) {
    for (let cx = 0; cx < w; cx++) {
      if (walk[cz * w + cx]) continue;
      for (let dz = -erodeCells; dz <= erodeCells; dz++) {
        for (let dx = -erodeCells; dx <= erodeCells; dx++) {
          if (dx * dx + dz * dz > erodeCells * erodeCells) continue;
          const nx = cx + dx, nz = cz + dz;
          if (nx < 0 || nz < 0 || nx >= w || nz >= h) continue;
          const ni = nz * w + nx;
          if (protect?.[ni] && dx * dx + dz * dz > 2) continue;
          eroded[ni] = 0;
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
  const spawnCells: number[] = [];
  const final = new Uint8Array(nCells);
  for (let ci = 0; ci < nCells; ci++) {
    if (eroded[ci] && comp[ci] === bestComp) { final[ci] = 1; spawnCells.push(ci); }
  }

  return {
    cell: CELL, minX, minZ, w, h,
    walkable: final, ground, spawn: new Uint32Array(spawnCells),
    meshMinY, climbHeight,
  };
}

// Synthesize a ground trimesh from the grid (for worlds without a mesh export)
// so Rapier raycasts (click-to-goal, camera clamp) and physics balls work.
export function gridToMesh(g: NavGrid): { positions: Float32Array; indices: Uint32Array } {
  const { w, h, ground } = g;
  const vertIdx = new Int32Array(w * h).fill(-1);
  const verts: number[] = [];
  for (let cz = 0; cz < h; cz++) {
    for (let cx = 0; cx < w; cx++) {
      const ci = cz * w + cx;
      if (Number.isNaN(ground[ci])) continue;
      vertIdx[ci] = verts.length / 3;
      verts.push(g.minX + (cx + 0.5) * g.cell, ground[ci], g.minZ + (cz + 0.5) * g.cell);
    }
  }
  const idx: number[] = [];
  for (let cz = 0; cz < h - 1; cz++) {
    for (let cx = 0; cx < w - 1; cx++) {
      const a = vertIdx[cz * w + cx], b = vertIdx[cz * w + cx + 1];
      const c = vertIdx[(cz + 1) * w + cx], d = vertIdx[(cz + 1) * w + cx + 1];
      if (a < 0 || b < 0 || c < 0 || d < 0) continue;
      idx.push(a, b, c, b, d, c);
    }
  }
  return { positions: new Float32Array(verts), indices: new Uint32Array(idx) };
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
  climbHeight: number;
  walkable: Uint8Array; ground: Float32Array; spawn: Uint32Array;
}

export function toTransfer(g: NavGrid): NavGridTransfer { return { ...g }; }
export function fromTransfer(t: NavGridTransfer): NavGrid { return { ...t }; }
