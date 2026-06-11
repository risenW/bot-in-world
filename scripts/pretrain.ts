// Headless pretraining in Node — same env + PPO code the browser runs.
// Produces public/checkpoints/pretrained.pfbt for the "Load pretrained" button.
//
// Usage: npm run pretrain [-- --level warehouse --steps 3000000 --seed 42]

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePly } from '../src/sim/ply';
import { bakeNavGrid } from '../src/sim/navgrid';
import { VecTrainer } from '../src/sim/vectrain';
import { encodeCheckpoint } from '../src/sim/checkpoint';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

const level = arg('level', 'warehouse');
const totalSteps = parseInt(arg('steps', '3000000'), 10);
const seed = parseInt(arg('seed', '42'), 10);
const out = arg('out', resolve(root, 'public/checkpoints/pretrained.pfbt'));

console.log(`[pretrain] level=${level} steps=${totalSteps} seed=${seed}`);
const meshPath = resolve(root, `public/levels/${level}/mesh_simplified.ply`);
const mesh = parsePly(readFileSync(meshPath).buffer as ArrayBuffer);
console.log(`[pretrain] mesh: ${mesh.positions.length / 3} verts, ${mesh.indices.length / 3} tris`);

const splatPath = resolve(root, `public/levels/${level}/world.ply`);
const splat = existsSync(splatPath)
  ? parsePly(readFileSync(splatPath).buffer as ArrayBuffer, { transform: false }).positions
  : undefined;
if (!splat) console.warn('[pretrain] no world.ply — spawn filter disabled');

const t0 = Date.now();
const grid = bakeNavGrid(mesh, splat);
console.log(`[pretrain] navgrid: ${grid.w}x${grid.h} cells, ${grid.spawn.length} walkable (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
if (grid.spawn.length < 200) {
  console.error('[pretrain] walkable area too small — check the level');
  process.exit(1);
}

const trainer = new VecTrainer(grid, seed);
const start = Date.now();
let lastLog = 0;
while (trainer.globalStep < totalSteps) {
  const progress = trainer.globalStep / totalSteps;
  const lrScale = 1 - 0.9 * progress;
  const s = trainer.iterate(lrScale);
  if (s.update - lastLog >= 10) {
    lastLog = s.update;
    const elapsed = (Date.now() - start) / 1000;
    console.log(
      `[pretrain] step=${(s.globalStep / 1e6).toFixed(2)}M ` +
      `sps=${Math.round(s.globalStep / elapsed)} ` +
      `return=${s.meanReturn.toFixed(2)} success=${(s.successRate * 100).toFixed(0)}% ` +
      `epLen=${s.meanEpLen.toFixed(0)} goalDist<=${s.curriculum.toFixed(1)}m ` +
      `kl=${s.losses.approxKl.toFixed(4)} ent=${s.losses.entropy.toFixed(3)}`,
    );
  }
}

mkdirSync(dirname(out), { recursive: true });
const buf = encodeCheckpoint(trainer.policy, {
  trainedSteps: trainer.globalStep,
  world: level,
  created: new Date().toISOString(),
  notes: `pretrained headless in Node, seed=${seed}`,
});
writeFileSync(out, Buffer.from(buf));
console.log(`[pretrain] saved ${out} (${(buf.byteLength / 1024).toFixed(0)} KB) ` +
  `final success=${(trainer.successRate() * 100).toFixed(0)}%`);
