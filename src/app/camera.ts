// Orbit camera with optional bot-follow and optional collision clamping.

import * as pc from 'playcanvas';
import RAPIER from '@dimforge/rapier3d-compat';

export class OrbitCamera {
  entity: pc.Entity;
  yaw = 0.6;
  pitch = -0.3;
  distance = 3.4;
  private clampedDist = 3.4;
  target = new pc.Vec3(0, 1, 0);
  follow = true;
  collision = true; // clamp to room interior; generated worlds have low/odd ceilings

  private dragging = false;
  private lastX = 0; private lastY = 0;
  private lastDragTime = -Infinity;

  constructor(private app: pc.Application, private world: RAPIER.World, private levelCollider: () => RAPIER.Collider | undefined) {
    this.entity = new pc.Entity('camera');
    this.entity.addComponent('camera', {
      clearColor: new pc.Color(0.02, 0.03, 0.05),
      fov: 60,
      nearClip: 0.05,
      farClip: 250,
    });
    app.root.addChild(this.entity);

    const canvas = app.graphicsDevice.canvas as HTMLCanvasElement;
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.dragging = true;
      this.lastX = e.clientX; this.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX, dy = e.clientY - this.lastY;
      this.lastX = e.clientX; this.lastY = e.clientY;
      this.yaw -= dx * 0.005;
      this.pitch = Math.max(-1.45, Math.min(0.5, this.pitch - dy * 0.005));
      this.lastDragTime = performance.now();
    });
    canvas.addEventListener('pointerup', () => (this.dragging = false));
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance = Math.max(1.2, Math.min(40, this.distance * (1 + Math.sign(e.deltaY) * 0.09)));
    }, { passive: false });
  }

  setTarget(p: pc.Vec3, smooth: number): void {
    this.target.lerp(this.target, p, Math.min(1, smooth));
  }

  // Chase-cam: trail behind the bot's heading (the path behind it is open space,
  // so the collision clamp rarely kicks in). Suspended ~2s after a manual drag.
  chaseYaw(botYaw: number, dt: number): void {
    if (this.dragging || performance.now() - this.lastDragTime < 2000) return;
    const desired = botYaw + Math.PI;
    let d = desired - this.yaw;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    this.yaw += d * Math.min(1, dt * 1.6);
  }

  update(): void {
    const cp = this.pitch, cy = this.yaw;
    const dir = new pc.Vec3(
      Math.cos(cp) * Math.sin(cy),
      -Math.sin(cp),
      Math.cos(cp) * Math.cos(cy),
    );
    let dist = this.distance;
    const collider = this.levelCollider();
    if (this.collision && collider) {
      const origin = { x: this.target.x, y: this.target.y, z: this.target.z };
      const rdir = { x: dir.x, y: dir.y, z: dir.z };
      const ray = new RAPIER.Ray(origin, rdir);
      const hit = this.world.castRay(ray, dist, true, undefined, undefined, undefined, undefined,
        (c) => c === collider);
      if (hit) dist = Math.max(0.8, hit.timeOfImpact - 0.15);
    }
    // smooth zoom changes from clamping so the camera doesn't pop
    this.clampedDist += (dist - this.clampedDist) * 0.25;
    dist = Math.min(this.clampedDist, dist);
    const pos = new pc.Vec3(
      this.target.x + dir.x * dist,
      this.target.y + dir.y * dist,
      this.target.z + dir.z * dist,
    );
    this.entity.setPosition(pos);
    this.entity.lookAt(this.target);
  }

  // Screen ray for click-to-set-goal: returns world-space ray
  screenRay(x: number, y: number): { origin: pc.Vec3; dir: pc.Vec3 } {
    const cam = this.entity.camera!;
    const canvas = this.app.graphicsDevice.canvas as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    const from = cam.screenToWorld(x - rect.left, y - rect.top, cam.nearClip);
    const to = cam.screenToWorld(x - rect.left, y - rect.top, cam.farClip);
    const dir = new pc.Vec3().sub2(to, from).normalize();
    return { origin: from, dir };
  }
}
