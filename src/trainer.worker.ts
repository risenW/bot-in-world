// Training web worker: owns the VecTrainer, streams stats + weights to the UI.

import { fromTransfer, NavGridTransfer } from './sim/navgrid';
import { VecTrainer, TrainStats } from './sim/vectrain';

export interface WorkerInMsg {
  type: 'init' | 'start' | 'pause' | 'loadWeights' | 'reset';
  grid?: NavGridTransfer;
  seed?: number;
  weights?: Float32Array;
  trainedSteps?: number;
}

export interface WorkerOutMsg {
  type: 'ready' | 'stats' | 'weights';
  stats?: TrainStats;
  weights?: Float32Array;
  trainedSteps?: number;
}

let trainer: VecTrainer | null = null;
let running = false;
let baseSteps = 0; // steps from a loaded checkpoint

const TOTAL_STEPS_FOR_ANNEAL = 20_000_000;
const MIN_LR_RATIO = 0.1;

function post(msg: WorkerOutMsg, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

function loop(): void {
  if (!trainer || !running) return;
  const progress = Math.min(1, (baseSteps + trainer.globalStep) / TOTAL_STEPS_FOR_ANNEAL);
  const lrScale = 1 - (1 - MIN_LR_RATIO) * progress;
  const stats = trainer.iterate(lrScale);
  post({ type: 'stats', stats });
  const weights = trainer.policy.serialize();
  post({ type: 'weights', weights, trainedSteps: baseSteps + trainer.globalStep }, [weights.buffer]);
  // setTimeout(0) keeps the worker responsive to pause/load messages
  setTimeout(loop, 0);
}

self.onmessage = (ev: MessageEvent<WorkerInMsg>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'init': {
      trainer = new VecTrainer(fromTransfer(msg.grid!), msg.seed ?? 42);
      running = false;
      baseSteps = 0;
      post({ type: 'ready' });
      // surface the initial (random) policy so the showcase bot moves immediately
      const weights = trainer.policy.serialize();
      post({ type: 'weights', weights, trainedSteps: 0 }, [weights.buffer]);
      break;
    }
    case 'start':
      if (trainer && !running) { running = true; loop(); }
      break;
    case 'pause':
      running = false;
      break;
    case 'loadWeights':
      if (trainer && msg.weights) {
        trainer.loadWeights(msg.weights);
        baseSteps = msg.trainedSteps ?? 0;
        trainer.globalStep = 0;
        const weights = trainer.policy.serialize();
        post({ type: 'weights', weights, trainedSteps: baseSteps }, [weights.buffer]);
      }
      break;
    case 'reset':
      running = false;
      trainer = null;
      break;
  }
};
