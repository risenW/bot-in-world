// PPO + MLP policy in plain TypeScript.
//
// This is a faithful port of PufferLib's PPO trainer ("PuffeRL", puffer.ai,
// MIT license) to TypeScript so training can run 100% client-side in a web
// worker. Architecture mirrors puffernet's MLP policy (shared trunk + actor
// + value heads) and the weight checkpoint is puffernet-style: all layers
// flattened and concatenated in PyTorch order ([out,in] row-major weight,
// then bias). Defaults derive from PufferLib's config/default.ini, adjusted
// for a plain Adam MLP (PufferLib's 0.015 lr pairs with MinGRU+Muon).

import { Rng, gaussian } from './rng';

export interface PpoConfig {
  obsSize: number;
  hidden: number;
  actions: number;
  numEnvs: number;
  horizon: number;        // PufferLib default.ini: horizon = 64
  epochs: number;
  minibatch: number;
  lr: number;
  gamma: number;          // PufferLib: 0.995 — we use 0.99 for short episodes
  gaeLambda: number;      // PufferLib: 0.90
  clipCoef: number;       // PufferLib: 0.2
  vfCoef: number;
  vfClipCoef: number;     // PufferLib: 0.2
  entCoef: number;        // PufferLib: 0.001
  maxGradNorm: number;    // PufferLib: 1.5
  adamBeta1: number;      // PufferLib: 0.95
  adamBeta2: number;      // PufferLib: 0.999
  adamEps: number;
}

export const DEFAULT_PPO: Omit<PpoConfig, 'obsSize' | 'actions'> = {
  hidden: 128,
  numEnvs: 64,
  horizon: 64,
  epochs: 3,
  minibatch: 2048,
  lr: 1e-3,
  gamma: 0.99,
  gaeLambda: 0.90,
  clipCoef: 0.2,
  vfCoef: 0.5,
  vfClipCoef: 0.2,
  entCoef: 0.005,
  maxGradNorm: 1.5,
  adamBeta1: 0.95,
  adamBeta2: 0.999,
  adamEps: 1e-8,
};

// ---------------- network ----------------
// obs -> fc1(h) relu -> fc2(h) relu -> { actor(actions), value(1) }

export interface LayerSpec { name: string; rows: number; cols: number }

export class Policy {
  obsSize: number; hidden: number; actions: number;
  w1: Float32Array; b1: Float32Array;
  w2: Float32Array; b2: Float32Array;
  wa: Float32Array; ba: Float32Array;
  wv: Float32Array; bv: Float32Array;

  constructor(obsSize: number, hidden: number, actions: number, rng?: Rng) {
    this.obsSize = obsSize; this.hidden = hidden; this.actions = actions;
    this.w1 = new Float32Array(hidden * obsSize);
    this.b1 = new Float32Array(hidden);
    this.w2 = new Float32Array(hidden * hidden);
    this.b2 = new Float32Array(hidden);
    this.wa = new Float32Array(actions * hidden);
    this.ba = new Float32Array(actions);
    this.wv = new Float32Array(hidden);
    this.bv = new Float32Array(1);
    if (rng) this.init(rng);
  }

  init(rng: Rng): void {
    const he = (arr: Float32Array, fanIn: number, scale = 1) => {
      const s = scale * Math.sqrt(2 / fanIn);
      for (let i = 0; i < arr.length; i++) arr[i] = gaussian(rng) * s;
    };
    he(this.w1, this.obsSize);
    he(this.w2, this.hidden);
    he(this.wa, this.hidden, 0.01); // near-uniform initial policy
    he(this.wv, this.hidden, 1.0);
  }

  layout(): LayerSpec[] {
    return [
      { name: 'fc1.weight', rows: this.hidden, cols: this.obsSize },
      { name: 'fc1.bias', rows: this.hidden, cols: 1 },
      { name: 'fc2.weight', rows: this.hidden, cols: this.hidden },
      { name: 'fc2.bias', rows: this.hidden, cols: 1 },
      { name: 'actor.weight', rows: this.actions, cols: this.hidden },
      { name: 'actor.bias', rows: this.actions, cols: 1 },
      { name: 'value.weight', rows: 1, cols: this.hidden },
      { name: 'value.bias', rows: 1, cols: 1 },
    ];
  }

  params(): Float32Array[] {
    return [this.w1, this.b1, this.w2, this.b2, this.wa, this.ba, this.wv, this.bv];
  }

  // puffernet-style flat export: concat in layout order
  serialize(): Float32Array {
    const parts = this.params();
    const total = parts.reduce((s, p) => s + p.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  load(flat: Float32Array): void {
    const parts = this.params();
    let o = 0;
    for (const p of parts) { p.set(flat.subarray(o, o + p.length)); o += p.length; }
  }

  // Forward for a batch. Buffers sized for batch n. Returns logits/values and
  // keeps activations for backward.
  forward(obs: Float32Array, n: number, acts: Activations): void {
    const { obsSize: O, hidden: H, actions: A } = this;
    const { h1, h2, logits, values } = acts;
    for (let b = 0; b < n; b++) {
      const ob = b * O;
      for (let j = 0; j < H; j++) {
        let s = this.b1[j];
        const wRow = j * O;
        for (let i = 0; i < O; i++) s += obs[ob + i] * this.w1[wRow + i];
        h1[b * H + j] = s > 0 ? s : 0;
      }
      for (let j = 0; j < H; j++) {
        let s = this.b2[j];
        const wRow = j * H;
        const hb = b * H;
        for (let i = 0; i < H; i++) s += h1[hb + i] * this.w2[wRow + i];
        h2[b * H + j] = s > 0 ? s : 0;
      }
      const hb = b * H;
      for (let a = 0; a < A; a++) {
        let s = this.ba[a];
        const wRow = a * H;
        for (let i = 0; i < H; i++) s += h2[hb + i] * this.wa[wRow + i];
        logits[b * A + a] = s;
      }
      let v = this.bv[0];
      for (let i = 0; i < H; i++) v += h2[hb + i] * this.wv[i];
      values[b] = v;
    }
  }
}

export interface Activations {
  h1: Float32Array; h2: Float32Array;
  logits: Float32Array; values: Float32Array;
}

export function makeActivations(p: Policy, n: number): Activations {
  return {
    h1: new Float32Array(n * p.hidden),
    h2: new Float32Array(n * p.hidden),
    logits: new Float32Array(n * p.actions),
    values: new Float32Array(n),
  };
}

// softmax in place over logits row; returns logsumexp
export function softmaxRow(logits: Float32Array, off: number, n: number, probs: Float32Array, pOff: number): void {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (logits[off + i] > mx) mx = logits[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(logits[off + i] - mx); probs[pOff + i] = e; sum += e; }
  for (let i = 0; i < n; i++) probs[pOff + i] /= sum;
}

export function sampleCategorical(probs: Float32Array, off: number, n: number, u: number): number {
  let acc = 0;
  for (let i = 0; i < n; i++) { acc += probs[off + i]; if (u < acc) return i; }
  return n - 1;
}

export function argmaxRow(arr: Float32Array, off: number, n: number): number {
  let best = 0, bv = -Infinity;
  for (let i = 0; i < n; i++) if (arr[off + i] > bv) { bv = arr[off + i]; best = i; }
  return best;
}

// ---------------- Adam ----------------

export class Adam {
  m: Float32Array[]; v: Float32Array[]; t = 0;
  constructor(params: Float32Array[], private cfg: PpoConfig) {
    this.m = params.map((p) => new Float32Array(p.length));
    this.v = params.map((p) => new Float32Array(p.length));
  }
  step(params: Float32Array[], grads: Float32Array[], lr: number): void {
    this.t++;
    const { adamBeta1: b1, adamBeta2: b2, adamEps: eps } = this.cfg;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    for (let k = 0; k < params.length; k++) {
      const p = params[k], g = grads[k], m = this.m[k], v = this.v[k];
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
  }
}

// ---------------- PPO update ----------------

export interface Rollout {
  obs: Float32Array;        // [steps, obsSize]
  actions: Int32Array;      // [steps]
  logprobs: Float32Array;   // [steps]
  values: Float32Array;     // [steps]
  rewards: Float32Array;    // [steps]
  dones: Uint8Array;        // [steps] done AFTER this transition
  advantages: Float32Array; // [steps]
  returns: Float32Array;    // [steps]
}

export function makeRollout(steps: number, obsSize: number): Rollout {
  return {
    obs: new Float32Array(steps * obsSize),
    actions: new Int32Array(steps),
    logprobs: new Float32Array(steps),
    values: new Float32Array(steps),
    rewards: new Float32Array(steps),
    dones: new Uint8Array(steps),
    advantages: new Float32Array(steps),
    returns: new Float32Array(steps),
  };
}

// GAE over [numEnvs, horizon] layout: index = t*numEnvs + e
export function computeGae(r: Rollout, lastValues: Float32Array, numEnvs: number, horizon: number, gamma: number, lambda: number): void {
  for (let e = 0; e < numEnvs; e++) {
    let lastGae = 0;
    let nextValue = lastValues[e];
    let nextNonTerminal = r.dones[(horizon - 1) * numEnvs + e] ? 0 : 1;
    for (let t = horizon - 1; t >= 0; t--) {
      const i = t * numEnvs + e;
      const nonTerminal = t === horizon - 1 ? nextNonTerminal : (r.dones[i] ? 0 : 1);
      const delta = r.rewards[i] + gamma * nextValue * nonTerminal - r.values[i];
      lastGae = delta + gamma * lambda * nonTerminal * lastGae;
      r.advantages[i] = lastGae;
      r.returns[i] = lastGae + r.values[i];
      nextValue = r.values[i];
    }
  }
}

export interface UpdateStats {
  policyLoss: number; valueLoss: number; entropy: number; approxKl: number; clipFrac: number;
}

export class PpoTrainer {
  policy: Policy;
  adam: Adam;
  cfg: PpoConfig;
  grads: Float32Array[];
  acts: Activations;
  probs: Float32Array;
  dLogits: Float32Array;
  dH2: Float32Array;
  dH1: Float32Array;
  rng: Rng;

  constructor(policy: Policy, cfg: PpoConfig, rng: Rng) {
    this.policy = policy;
    this.cfg = cfg;
    this.rng = rng;
    this.adam = new Adam(policy.params(), cfg);
    this.grads = policy.params().map((p) => new Float32Array(p.length));
    const mb = cfg.minibatch;
    this.acts = makeActivations(policy, mb);
    this.probs = new Float32Array(mb * policy.actions);
    this.dLogits = new Float32Array(mb * policy.actions);
    this.dH2 = new Float32Array(mb * policy.hidden);
    this.dH1 = new Float32Array(mb * policy.hidden);
  }

  update(roll: Rollout, lr: number): UpdateStats {
    const { cfg, policy } = this;
    const steps = roll.actions.length;
    const idx = new Int32Array(steps);
    for (let i = 0; i < steps; i++) idx[i] = i;

    // normalize advantages (global)
    let mean = 0;
    for (let i = 0; i < steps; i++) mean += roll.advantages[i];
    mean /= steps;
    let varSum = 0;
    for (let i = 0; i < steps; i++) { const d = roll.advantages[i] - mean; varSum += d * d; }
    const std = Math.sqrt(varSum / steps) + 1e-8;
    for (let i = 0; i < steps; i++) roll.advantages[i] = (roll.advantages[i] - mean) / std;

    const stats: UpdateStats = { policyLoss: 0, valueLoss: 0, entropy: 0, approxKl: 0, clipFrac: 0 };
    let statN = 0;

    const O = policy.obsSize, A = policy.actions;
    const mbObs = new Float32Array(cfg.minibatch * O);

    for (let epoch = 0; epoch < cfg.epochs; epoch++) {
      // shuffle
      for (let i = steps - 1; i > 0; i--) {
        const j = (this.rng() * (i + 1)) | 0;
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }
      for (let start = 0; start + cfg.minibatch <= steps; start += cfg.minibatch) {
        const n = cfg.minibatch;
        for (let b = 0; b < n; b++) {
          const s = idx[start + b];
          mbObs.set(roll.obs.subarray(s * O, s * O + O), b * O);
        }
        const st = this.minibatchStep(mbObs, idx, start, n, roll, lr);
        stats.policyLoss += st.policyLoss; stats.valueLoss += st.valueLoss;
        stats.entropy += st.entropy; stats.approxKl += st.approxKl; stats.clipFrac += st.clipFrac;
        statN++;
      }
    }
    if (statN > 0) {
      stats.policyLoss /= statN; stats.valueLoss /= statN; stats.entropy /= statN;
      stats.approxKl /= statN; stats.clipFrac /= statN;
    }
    return stats;
  }

  private minibatchStep(mbObs: Float32Array, idx: Int32Array, start: number, n: number, roll: Rollout, lr: number): UpdateStats {
    const { policy, cfg, acts, probs, dLogits, dH2, dH1, grads } = this;
    const A = policy.actions, H = policy.hidden, O = policy.obsSize;

    policy.forward(mbObs, n, acts);
    for (const g of grads) g.fill(0);

    let policyLossAcc = 0, valueLossAcc = 0, entropyAcc = 0, klAcc = 0, clipCount = 0;

    for (let b = 0; b < n; b++) {
      const s = idx[start + b];
      softmaxRow(acts.logits, b * A, A, probs, b * A);
      const action = roll.actions[s];
      const p = Math.max(probs[b * A + action], 1e-10);
      const newLogprob = Math.log(p);
      const oldLogprob = roll.logprobs[s];
      const logratio = newLogprob - oldLogprob;
      const ratio = Math.exp(logratio);
      const adv = roll.advantages[s];

      klAcc += (ratio - 1) - logratio;
      if (Math.abs(ratio - 1) > cfg.clipCoef) clipCount++;

      // pg loss: max(-adv*ratio, -adv*clip(ratio))
      const unclipped = -adv * ratio;
      const clipped = -adv * Math.min(Math.max(ratio, 1 - cfg.clipCoef), 1 + cfg.clipCoef);
      const useUnclipped = unclipped >= clipped;
      policyLossAcc += Math.max(unclipped, clipped);

      // d(pgLoss)/d(logits): only unclipped branch has gradient
      // d(-adv*ratio)/dlogit_k = -adv*ratio*(1[k=a] - p_k)
      const gScale = useUnclipped ? -adv * ratio : 0;

      // entropy: H = -sum p log p ; d(-entCoef*H)/dlogit_k = -entCoef * (-p_k*(log p_k + H... ))
      let ent = 0;
      for (let k = 0; k < A; k++) {
        const pk = Math.max(probs[b * A + k], 1e-10);
        ent -= pk * Math.log(pk);
      }
      entropyAcc += ent;

      for (let k = 0; k < A; k++) {
        const pk = Math.max(probs[b * A + k], 1e-10);
        const onehot = k === action ? 1 : 0;
        const dPg = gScale * (onehot - pk);
        // dH/dlogit_k = -p_k * (log p_k + ent)  →  d(-c*H)/dlogit_k = c * p_k * (log p_k + ent)
        const dEnt = cfg.entCoef * pk * (Math.log(pk) + ent);
        dLogits[b * A + k] = (dPg + dEnt) / n;
      }

      // value loss (clipped, PufferLib-style)
      const v = acts.values[b];
      const oldV = roll.values[s];
      const ret = roll.returns[s];
      const vClipped = oldV + Math.min(Math.max(v - oldV, -cfg.vfClipCoef), cfg.vfClipCoef);
      const lossUncl = (v - ret) ** 2;
      const lossClip = (vClipped - ret) ** 2;
      const useUncl = lossUncl >= lossClip;
      valueLossAcc += 0.5 * Math.max(lossUncl, lossClip);
      const insideClip = Math.abs(v - oldV) < cfg.vfClipCoef;
      const dV = cfg.vfCoef * ((useUncl ? (v - ret) : (insideClip ? (vClipped - ret) : 0))) / n;

      // backward through heads into dH2
      const hb = b * H;
      for (let i = 0; i < H; i++) {
        let s2 = dV * policy.wv[i];
        for (let k = 0; k < A; k++) s2 += dLogits[b * A + k] * policy.wa[k * H + i];
        dH2[hb + i] = acts.h2[hb + i] > 0 ? s2 : 0;
      }
      // grads for heads
      for (let k = 0; k < A; k++) {
        const dl = dLogits[b * A + k];
        if (dl !== 0) {
          const row = k * H;
          for (let i = 0; i < H; i++) grads[4][row + i] += dl * acts.h2[hb + i];
          grads[5][k] += dl;
        }
      }
      for (let i = 0; i < H; i++) grads[6][i] += dV * acts.h2[hb + i];
      grads[7][0] += dV;

      // backward fc2 -> dH1
      for (let i = 0; i < H; i++) {
        let s1 = 0;
        for (let j = 0; j < H; j++) s1 += dH2[hb + j] * policy.w2[j * H + i];
        dH1[hb + i] = acts.h1[hb + i] > 0 ? s1 : 0;
      }
      for (let j = 0; j < H; j++) {
        const d = dH2[hb + j];
        if (d !== 0) {
          const row = j * H;
          for (let i = 0; i < H; i++) grads[2][row + i] += d * acts.h1[hb + i];
          grads[3][j] += d;
        }
      }
      // backward fc1
      const ob = b * O;
      for (let j = 0; j < H; j++) {
        const d = dH1[hb + j];
        if (d !== 0) {
          const row = j * O;
          for (let i = 0; i < O; i++) grads[0][row + i] += d * mbObs[ob + i];
          grads[1][j] += d;
        }
      }
    }

    // grad clip (global norm)
    let normSq = 0;
    for (const g of grads) for (let i = 0; i < g.length; i++) normSq += g[i] * g[i];
    const norm = Math.sqrt(normSq);
    if (norm > cfg.maxGradNorm) {
      const scale = cfg.maxGradNorm / (norm + 1e-8);
      for (const g of grads) for (let i = 0; i < g.length; i++) g[i] *= scale;
    }

    this.adam.step(policy.params(), grads, lr);

    return {
      policyLoss: policyLossAcc / n,
      valueLoss: valueLossAcc / n,
      entropy: entropyAcc / n,
      approxKl: klAcc / n,
      clipFrac: clipCount / n,
    };
  }
}
