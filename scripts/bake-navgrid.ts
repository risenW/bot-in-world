// Inspect a level's navgrid + write real mesh bounds into its manifest.
// Usage: npm run bake [-- --level warehouse]

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePly } from '../src/sim/ply';
import { bakeNavGrid } from '../src/sim/navgrid';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const i = process.argv.indexOf('--level');
const level = i >= 0 ? process.argv[i + 1] : 'warehouse';

const dir = resolve(root, `public/levels/${level}`);
const mesh = parsePly(readFileSync(resolve(dir, 'mesh_simplified.ply')).buffer as ArrayBuffer);
const grid = bakeNavGrid(mesh);

const walkablePct = (100 * grid.spawn.length) / (grid.w * grid.h);
console.log(`level=${level}`);
console.log(`  mesh: ${mesh.positions.length / 3} verts, ${mesh.indices.length / 3} tris`);
console.log(`  aabb: min=[${mesh.aabb.min.map((v) => v.toFixed(2))}] max=[${mesh.aabb.max.map((v) => v.toFixed(2))}]`);
console.log(`  grid: ${grid.w}x${grid.h} @ ${grid.cell}m`);
console.log(`  walkable (largest region): ${grid.spawn.length} cells (${walkablePct.toFixed(1)}% of grid, ` +
  `${(grid.spawn.length * grid.cell * grid.cell).toFixed(1)} m²)`);

const manifestPath = resolve(dir, 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
manifest.bounds = { min: mesh.aabb.min, max: mesh.aabb.max };
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
console.log(`  manifest bounds updated`);
