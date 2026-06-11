// Checkpoint container: JSON meta + puffernet-style flat float32 weights.
// Layout: 'PFBT' magic | u32 version | u32 jsonByteLength | json (utf8) | f32 weights[]

import { Policy, LayerSpec } from './ppo';

export interface CheckpointMeta {
  format: 'puffernet-flat';
  arch: { obsSize: number; hidden: number; actions: number };
  layout: LayerSpec[];
  trainedSteps: number;
  world: string;
  created: string;
  notes?: string;
}

const MAGIC = 0x54424650; // 'PFBT' LE

export function encodeCheckpoint(policy: Policy, meta: Omit<CheckpointMeta, 'format' | 'arch' | 'layout'>): ArrayBuffer {
  const weights = policy.serialize();
  const fullMeta: CheckpointMeta = {
    format: 'puffernet-flat',
    arch: { obsSize: policy.obsSize, hidden: policy.hidden, actions: policy.actions },
    layout: policy.layout(),
    ...meta,
  };
  const json = new TextEncoder().encode(JSON.stringify(fullMeta));
  const jsonPadded = (json.length + 3) & ~3; // 4-byte align weights
  const buf = new ArrayBuffer(12 + jsonPadded + weights.length * 4);
  const view = new DataView(buf);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, 1, true);
  view.setUint32(8, json.length, true);
  new Uint8Array(buf, 12, json.length).set(json);
  new Float32Array(buf, 12 + jsonPadded, weights.length).set(weights);
  return buf;
}

export function decodeCheckpoint(buf: ArrayBuffer): { meta: CheckpointMeta; weights: Float32Array } {
  const view = new DataView(buf);
  if (view.getUint32(0, true) !== MAGIC) throw new Error('Not a PufferBot checkpoint (.pfbt)');
  const jsonLen = view.getUint32(8, true);
  const jsonPadded = (jsonLen + 3) & ~3;
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, jsonLen))) as CheckpointMeta;
  const weights = new Float32Array(buf.slice(12 + jsonPadded));
  return { meta, weights };
}

export function policyFromCheckpoint(buf: ArrayBuffer): { policy: Policy; meta: CheckpointMeta } {
  const { meta, weights } = decodeCheckpoint(buf);
  const policy = new Policy(meta.arch.obsSize, meta.arch.hidden, meta.arch.actions);
  policy.load(weights);
  return { policy, meta };
}
