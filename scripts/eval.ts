// Evaluate a checkpoint on a level (greedy policy, random spawn/goal pairs).
// This is the generalization test: train on one world, eval on another.
// Usage: npx tsx scripts/eval.ts --level gallery --ckpt public/checkpoints/pretrained.pfbt --episodes 200

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePly } from '../src/sim/ply';
import { bakeNavGrid } from '../src/sim/navgrid';
import { NavEnv, OBS_SIZE, N_ACTIONS } from '../src/sim/env';
import { makeActivations, argmaxRow } from '../src/sim/ppo';
import { policyFromCheckpoint } from '../src/sim/checkpoint';
import { mulberry32 } from '../src/sim/rng';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

const level = arg('level', 'gallery');
const ckptPath = arg('ckpt', resolve(root, 'public/checkpoints/pretrained.pfbt'));
const episodes = parseInt(arg('episodes', '200'), 10);
const seed = parseInt(arg('seed', '7'), 10);

const mesh = parsePly(readFileSync(resolve(root, `public/levels/${level}/mesh_simplified.ply`)).buffer as ArrayBuffer);
const splatPath = resolve(root, `public/levels/${level}/world.ply`);
const splat = existsSync(splatPath)
  ? parsePly(readFileSync(splatPath).buffer as ArrayBuffer, { transform: false }).positions
  : undefined;
const grid = bakeNavGrid(mesh, splat);
const { policy, meta } = policyFromCheckpoint(readFileSync(ckptPath).buffer as ArrayBuffer);
console.log(`[eval] checkpoint: trained ${(meta.trainedSteps / 1e6).toFixed(1)}M steps on "${meta.world}" — evaluating on "${level}"`);

const rng = mulberry32(seed);
const env = new NavEnv(grid, rng);
env.goalDistMin = 2;
env.goalDistMax = 10;
const acts = makeActivations(policy, 1);
const obs = new Float32Array(OBS_SIZE);

let successes = 0, totalLen = 0, totalDist = 0;
for (let ep = 0; ep < episodes; ep++) {
  env.reset();
  totalDist += env.goalDist();
  for (;;) {
    env.observe(obs, 0);
    policy.forward(obs, 1, acts);
    const res = env.step(argmaxRow(acts.logits, 0, N_ACTIONS));
    if (res.done) {
      if (res.success) successes++;
      totalLen += env.steps;
      break;
    }
  }
}
console.log(`[eval] ${level}: success ${successes}/${episodes} (${((100 * successes) / episodes).toFixed(1)}%), ` +
  `avg episode ${(totalLen / episodes).toFixed(0)} steps, avg goal dist ${(totalDist / episodes).toFixed(1)}m`);
