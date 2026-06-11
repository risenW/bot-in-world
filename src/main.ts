// PufferBot — a humanoid bot learns to navigate AI-generated 3D worlds,
// trained with a TypeScript port of PufferLib's PPO, live in your browser.

import * as pc from 'playcanvas';
import RAPIER from '@dimforge/rapier3d-compat';
import { Level, ViewMode } from './app/level';
import { Showcase } from './app/showcase';
import { OrbitCamera } from './app/camera';
import { Ui } from './app/ui';
import { toTransfer, groundHeight, isWalkable } from './sim/navgrid';
import { encodeCheckpoint, decodeCheckpoint } from './sim/checkpoint';
import { TrainStats } from './sim/vectrain';
import type { WorkerInMsg, WorkerOutMsg } from './trainer.worker';

const AUTOSAVE_KEY = 'pufferbot-autosave-v1';
const BALL_COLORS = [
  new pc.Color(0.937, 0.325, 0.314), new pc.Color(0.259, 0.647, 0.961),
  new pc.Color(0.4, 0.733, 0.416), new pc.Color(1.0, 0.933, 0.345),
  new pc.Color(0.671, 0.278, 0.737),
];

async function boot() {
  const setLoading = (step: string, frac: number) => {
    (document.getElementById('loading-fill') as HTMLElement).style.width = `${Math.round(frac * 100)}%`;
    (document.getElementById('loading-step') as HTMLElement).textContent = step;
  };
  setLoading('Initializing physics…', 0.02);
  await RAPIER.init();
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

  const canvas = document.getElementById('app-canvas') as HTMLCanvasElement;
  const app = new pc.Application(canvas, {
    mouse: new pc.Mouse(canvas),
    touch: new pc.TouchDevice(canvas),
    keyboard: new pc.Keyboard(window),
    graphicsDeviceOptions: { antialias: true },
  });
  app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  window.addEventListener('resize', () => app.resizeCanvas());
  app.start();

  // lights (the splat is unlit; these light the bot + markers)
  const sun = new pc.Entity('sun');
  sun.addComponent('light', { type: 'directional', intensity: 1.4, castShadows: false });
  sun.setEulerAngles(50, 30, 0);
  app.root.addChild(sun);
  app.scene.ambientLight = new pc.Color(0.45, 0.47, 0.55);

  // level
  const level = new Level(app, world);
  await level.load('warehouse', setLoading);

  // camera + showcase bot
  const camera = new OrbitCamera(app, world, () => level.levelCollider);
  const showcase = new Showcase(app, level.grid);
  camera.target.set(showcase.env.x, 1.2, showcase.env.z);

  // ---------------- training worker ----------------
  const worker = new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' });
  let latestWeights: Float32Array | null = null;
  let latestTrainedSteps = 0;
  let lastStats: TrainStats | null = null;
  let lastAutosave = 0;

  const workerSend = (msg: WorkerInMsg) => worker.postMessage(msg);
  const initWorker = () => workerSend({ type: 'init', grid: toTransfer(level.grid), seed: (Math.random() * 1e9) | 0 });

  worker.onmessage = (ev: MessageEvent<WorkerOutMsg>) => {
    const msg = ev.data;
    if (msg.type === 'stats' && msg.stats) {
      lastStats = msg.stats;
      ui.updateStats(msg.stats, latestTrainedSteps);
    } else if (msg.type === 'weights' && msg.weights) {
      latestWeights = msg.weights;
      latestTrainedSteps = msg.trainedSteps ?? latestTrainedSteps;
      showcase.setWeights(msg.weights);
      const now = performance.now();
      if (ui.training && now - lastAutosave > 15000) {
        lastAutosave = now;
        autosave();
      }
    }
  };

  function autosave() {
    if (!latestWeights) return;
    try {
      const buf = encodeCheckpoint(showcase.policy, {
        trainedSteps: latestTrainedSteps,
        world: level.manifest.id,
        created: new Date().toISOString(),
        notes: 'browser autosave',
      });
      localStorage.setItem(AUTOSAVE_KEY, bufToB64(buf));
    } catch (e) {
      console.warn('autosave failed', e);
    }
  }

  function applyCheckpoint(buf: ArrayBuffer, label: string) {
    try {
      const { meta, weights } = decodeCheckpoint(buf);
      showcase.setWeights(weights);
      latestWeights = weights;
      latestTrainedSteps = meta.trainedSteps;
      workerSend({ type: 'loadWeights', weights, trainedSteps: meta.trainedSteps });
      ui.toast(`${label}: ${(meta.trainedSteps / 1e6).toFixed(1)}M steps (${meta.world})`);
    } catch (e) {
      ui.toast(`Failed to load checkpoint: ${(e as Error).message}`);
    }
  }

  // ---------------- balls ----------------
  const balls: { body: RAPIER.RigidBody; entity: pc.Entity }[] = [];
  function spawnBall(colorIndex: number) {
    const r = 0.16;
    const t = camera.target;
    const x = t.x + (Math.random() - 0.5) * 0.8;
    const z = t.z + (Math.random() - 0.5) * 0.8;
    const y = (groundHeight(level.grid, x, z) || t.y) + 2.2;
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(x, y, z).setLinearDamping(0.3).setAngularDamping(0.4),
    );
    world.createCollider(RAPIER.ColliderDesc.ball(r).setRestitution(0.55).setFriction(0.8), body);
    const material = new pc.StandardMaterial();
    material.diffuse = BALL_COLORS[colorIndex % BALL_COLORS.length];
    material.gloss = 0.7;
    material.update();
    const entity = new pc.Entity('ball');
    entity.addComponent('render', { type: 'sphere', material });
    entity.setLocalScale(r * 2, r * 2, r * 2);
    app.root.addChild(entity);
    balls.push({ body, entity });
    if (balls.length > 40) {
      const old = balls.shift()!;
      world.removeRigidBody(old.body);
      old.entity.destroy();
    }
  }

  // ---------------- world switching ----------------
  let switching = false;
  async function switchWorld(id: string) {
    if (switching || id === level.manifest.id) return;
    switching = true;
    const wasTraining = ui.training;
    workerSend({ type: 'pause' });
    ui.setTraining(false);
    const overlay = document.getElementById('loading')!;
    overlay.classList.remove('hidden');
    try {
      await level.load(id, setLoading);
      showcase.setGrid(level.grid);
      camera.target.set(showcase.env.x, 1.2, showcase.env.z);
      camera.distance = 4.2;
      // re-init trainer on the new grid, carrying weights over
      initWorker();
      if (latestWeights) workerSend({ type: 'loadWeights', weights: latestWeights.slice(), trainedSteps: latestTrainedSteps });
      ui.setViewMode(level.mode);
      if (id.startsWith('custom:')) ui.addCustomWorld(id, level.manifest.name);
      else ui.setWorld(id);
      ui.toast(wasTraining
        ? `Switched to ${level.manifest.name} — training paused, policy carried over`
        : `Switched to ${level.manifest.name}`);
    } catch (e) {
      ui.toast(`Failed to load world: ${(e as Error).message}`);
    } finally {
      overlay.classList.add('hidden');
      switching = false;
    }
  }

  // ---------------- UI ----------------
  const ui = new Ui({
    onTrainToggle: () => {
      if (ui.training) { workerSend({ type: 'pause' }); ui.setTraining(false); autosave(); }
      else { workerSend({ type: 'start' }); ui.setTraining(true); }
    },
    onSaveCheckpoint: () => {
      if (!latestWeights) { ui.toast('Nothing to save yet — start learning first'); return; }
      const buf = encodeCheckpoint(showcase.policy, {
        trainedSteps: latestTrainedSteps,
        world: level.manifest.id,
        created: new Date().toISOString(),
      });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([buf], { type: 'application/octet-stream' }));
      a.download = `pufferbot-${level.manifest.id}-${(latestTrainedSteps / 1e6).toFixed(1)}M.pfbt`;
      a.click();
      URL.revokeObjectURL(a.href);
    },
    onLoadCheckpointFile: async (file) => applyCheckpoint(await file.arrayBuffer(), file.name),
    onLoadPretrained: async () => {
      try {
        const res = await fetch('/checkpoints/pretrained.pfbt');
        if (!res.ok) throw new Error('no bundled checkpoint');
        applyCheckpoint(await res.arrayBuffer(), 'Pretrained');
      } catch {
        ui.toast('No pretrained checkpoint bundled yet — run: npm run pretrain');
      }
    },
    onLoadAutosave: () => {
      const b64 = localStorage.getItem(AUTOSAVE_KEY);
      if (!b64) { ui.toast('No autosave found'); return; }
      applyCheckpoint(b64ToBuf(b64), 'Autosave');
    },
    onWorldChange: (id) => void switchWorld(id),
    onLoadCustomWorld: (link) => {
      const m = link.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i);
      if (!m) { ui.toast('Paste a Spaitial world link like https://app.spaitial.ai/worlds/<id>'); return; }
      void switchWorld(`custom:${m[1].toLowerCase()}`);
    },
    onNewGoal: () => { showcase.newGoal(); ui.toast('New random goal set'); },
    onViewMode: (mode: ViewMode) => { level.setMode(mode); ui.setViewMode(mode); },
    onToggleNav: () => ui.setToggle('nav', level.toggleNav()),
    onToggleCameraCollision: () => { camera.collision = !camera.collision; ui.setToggle('cam', camera.collision); },
    onToggleFollow: () => { camera.follow = !camera.follow; ui.setToggle('follow', camera.follow); },
    onToggleGreedy: () => { showcase.greedy = !showcase.greedy; ui.setToggle('greedy', showcase.greedy); },
    onRespawn: () => showcase.respawn(),
    onSpawnBall: spawnBall,
  });

  ui.setToggle('cam', camera.collision);

  showcase.onEpisodeEnd = (success) => {
    if (success && !ui.training) ui.toast('🎉 Goal reached');
  };

  initWorker();

  // offer autosave restore
  if (localStorage.getItem(AUTOSAVE_KEY)) ui.toast('Autosave found — “🕘 Autosave” restores your last policy');

  // ---------------- input ----------------
  const keys: Record<string, () => void> = {
    b: () => { level.setMode('both'); ui.setViewMode('both'); },
    m: () => { level.setMode('mesh-overlay'); ui.setViewMode('mesh-overlay'); },
    p: () => { level.setMode('splat-only'); ui.setViewMode('splat-only'); },
    o: () => { level.setMode('mesh-only'); ui.setViewMode('mesh-only'); },
    n: () => ui.setToggle('nav', level.toggleNav()),
    c: () => { camera.collision = !camera.collision; ui.setToggle('cam', camera.collision); },
    f: () => { camera.follow = !camera.follow; ui.setToggle('follow', camera.follow); },
    g: () => { showcase.greedy = !showcase.greedy; ui.setToggle('greedy', showcase.greedy); },
    r: () => showcase.respawn(),
    '1': () => spawnBall(0), '2': () => spawnBall(1), '3': () => spawnBall(2),
    '4': () => spawnBall(3), '5': () => spawnBall(4),
  };
  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'SELECT' || (e.target as HTMLElement).tagName === 'INPUT') return;
    if (e.key === ' ') { e.preventDefault(); ui.training ? (workerSend({ type: 'pause' }), ui.setTraining(false)) : (workerSend({ type: 'start' }), ui.setTraining(true)); return; }
    keys[e.key.toLowerCase()]?.();
  });

  // click (not drag) on floor sets the bot's goal
  let downX = 0, downY = 0, downT = 0;
  canvas.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; downT = performance.now(); });
  canvas.addEventListener('pointerup', (e) => {
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6 || performance.now() - downT > 400) return;
    const { origin, dir } = camera.screenRay(e.clientX, e.clientY);
    const ray = new RAPIER.Ray({ x: origin.x, y: origin.y, z: origin.z }, { x: dir.x, y: dir.y, z: dir.z });
    const hit = world.castRay(ray, 200, true, undefined, undefined, undefined, undefined,
      (c) => c === level.levelCollider);
    if (!hit) return;
    const px = origin.x + dir.x * hit.timeOfImpact;
    const pz = origin.z + dir.z * hit.timeOfImpact;
    if (showcase.clickGoal(px, pz)) ui.toast('Goal set — bot is on its way');
    else ui.toast('That spot is not walkable');
  });

  // ---------------- frame loop ----------------
  app.on('update', (dt: number) => {
    world.timestep = Math.min(dt, 1 / 30);
    world.step();
    for (const b of balls) {
      const t = b.body.translation();
      b.entity.setPosition(t.x, t.y, t.z);
      const q = b.body.rotation();
      b.entity.setRotation(q.x, q.y, q.z, q.w);
    }
    showcase.update(dt);
    if (camera.follow) {
      const p = showcase.bot.position;
      camera.target.lerp(camera.target, new pc.Vec3(p.x, p.y + 0.8, p.z), Math.min(1, dt * 5));
      camera.chaseYaw(showcase.env.yaw, dt);
    }
    camera.update();
  });

  document.getElementById('loading')!.classList.add('hidden');
  ui.toast('Press ▶ Start learning, or ⚡ Load pretrained to skip ahead');
}

function bufToB64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}
function b64ToBuf(b64: string): ArrayBuffer {
  const s = atob(b64);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return bytes.buffer;
}

boot().catch((e) => {
  console.error(e);
  (document.getElementById('loading-step') as HTMLElement).textContent = `Boot failed: ${e.message}`;
});
