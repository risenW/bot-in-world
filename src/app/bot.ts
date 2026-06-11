// Procedural humanoid bot built from PlayCanvas primitives, with a walk cycle
// driven by travel speed. ~1.5m tall, cyan/violet "puffer" palette.

import * as pc from 'playcanvas';

function mat(color: pc.Color, emissive?: pc.Color): pc.StandardMaterial {
  const m = new pc.StandardMaterial();
  m.diffuse = color;
  m.gloss = 0.6;
  m.metalness = 0.3;
  m.useMetalness = true;
  if (emissive) { m.emissive = emissive; m.emissiveIntensity = 1.6; }
  m.update();
  return m;
}

function capsule(name: string, material: pc.StandardMaterial, radius: number, height: number): pc.Entity {
  const e = new pc.Entity(name);
  e.addComponent('render', { type: 'capsule', material });
  e.setLocalScale(radius * 2, height, radius * 2);
  return e;
}

export class HumanoidBot {
  root: pc.Entity;          // world position at ground level
  body: pc.Entity;          // yaw + bob
  private thighL: pc.Entity; private thighR: pc.Entity;
  private armL: pc.Entity; private armR: pc.Entity;
  private torso: pc.Entity;
  private phase = 0;
  private bobBase = 0;

  // interpolation state
  private fromX = 0; private fromZ = 0; private fromYaw = 0;
  private toX = 0; private toZ = 0; private toYaw = 0;
  private speedVis = 0;

  constructor(app: pc.Application) {
    const dark = mat(new pc.Color(0.16, 0.18, 0.24));
    const cyan = mat(new pc.Color(0.2, 0.65, 0.85));
    const violet = mat(new pc.Color(0.45, 0.35, 0.85));
    const glow = mat(new pc.Color(0.05, 0.05, 0.08), new pc.Color(0.35, 0.9, 1.0));

    this.root = new pc.Entity('bot');
    this.body = new pc.Entity('bot-body');
    this.root.addChild(this.body);

    const legH = 0.62, torsoH = 0.55, headR = 0.14;
    this.bobBase = legH;

    // torso
    this.torso = capsule('torso', cyan, 0.17, torsoH + 0.12);
    this.torso.setLocalPosition(0, legH + torsoH / 2, 0);
    this.body.addChild(this.torso);

    // chest light
    const chest = new pc.Entity('chest');
    chest.addComponent('render', { type: 'sphere', material: glow });
    chest.setLocalScale(0.1, 0.1, 0.06);
    chest.setLocalPosition(0, legH + torsoH * 0.72, 0.13);
    this.body.addChild(chest);

    // head
    const head = new pc.Entity('head');
    head.addComponent('render', { type: 'sphere', material: dark });
    head.setLocalScale(headR * 2.2, headR * 2, headR * 2.2);
    head.setLocalPosition(0, legH + torsoH + headR + 0.09, 0);
    this.body.addChild(head);
    // visor (eyes)
    const visor = new pc.Entity('visor');
    visor.addComponent('render', { type: 'sphere', material: glow });
    visor.setLocalScale(0.18, 0.07, 0.1);
    visor.setLocalPosition(0, legH + torsoH + headR + 0.1, headR * 0.85);
    this.body.addChild(visor);

    // legs (pivot at hip)
    this.thighL = this.makeLimb('legL', violet, 0.085, legH, 0.105, legH, 0);
    this.thighR = this.makeLimb('legR', violet, 0.085, legH, -0.105, legH, 0);
    // arms (pivot at shoulder)
    const armH = 0.5;
    this.armL = this.makeLimb('armL', dark, 0.06, armH, 0.26, legH + torsoH * 0.92, 0);
    this.armR = this.makeLimb('armR', dark, 0.06, armH, -0.26, legH + torsoH * 0.92, 0);

    app.root.addChild(this.root);
  }

  // limb group whose origin is the joint; capsule hangs below it
  private makeLimb(name: string, material: pc.StandardMaterial, r: number, len: number, x: number, y: number, z: number): pc.Entity {
    const joint = new pc.Entity(name);
    joint.setLocalPosition(x, y, z);
    const seg = capsule(name + '-seg', material, r, len);
    seg.setLocalPosition(0, -len / 2, 0);
    joint.addChild(seg);
    this.body.addChild(joint);
    return joint;
  }

  // Called when the underlying sim state advances (10 Hz); render interpolates.
  setTarget(x: number, z: number, yaw: number): void {
    this.fromX = this.toX; this.fromZ = this.toZ; this.fromYaw = this.toYaw;
    this.toX = x; this.toZ = z; this.toYaw = yaw;
  }

  snapTo(x: number, z: number, yaw: number): void {
    this.fromX = this.toX = x; this.fromZ = this.toZ = z; this.fromYaw = this.toYaw = yaw;
  }

  // alpha: 0..1 interpolation between previous and current sim state
  render(alpha: number, groundY: number, dt: number): void {
    const x = this.fromX + (this.toX - this.fromX) * alpha;
    const z = this.fromZ + (this.toZ - this.fromZ) * alpha;
    let dyaw = this.toYaw - this.fromYaw;
    if (dyaw > Math.PI) dyaw -= Math.PI * 2;
    if (dyaw < -Math.PI) dyaw += Math.PI * 2;
    const yaw = this.fromYaw + dyaw * alpha;

    const prev = this.root.getPosition();
    const dist = Math.hypot(x - prev.x, z - prev.z);
    const speed = dt > 0 ? dist / dt : 0;
    this.speedVis += (speed - this.speedVis) * Math.min(1, dt * 10);

    this.root.setPosition(x, groundY, z);
    this.body.setEulerAngles(0, (yaw * 180) / Math.PI, 0);

    // walk cycle
    this.phase += this.speedVis * dt * 5.2;
    const swingAmp = Math.min(1, this.speedVis / 1.4) * 32;
    const s = Math.sin(this.phase) * swingAmp;
    this.thighL.setLocalEulerAngles(s, 0, 0);
    this.thighR.setLocalEulerAngles(-s, 0, 0);
    this.armL.setLocalEulerAngles(-s * 0.75, 0, 6);
    this.armR.setLocalEulerAngles(s * 0.75, 0, -6);
    const bob = Math.abs(Math.cos(this.phase)) * Math.min(1, this.speedVis / 1.4) * 0.035;
    this.body.setLocalPosition(0, bob, 0);
  }

  get position(): pc.Vec3 { return this.root.getPosition(); }
}

// Glowing goal marker: ring + light beacon
export class GoalMarker {
  root: pc.Entity;
  private ring: pc.Entity;
  private beam: pc.Entity;
  private t = 0;

  constructor(app: pc.Application) {
    this.root = new pc.Entity('goal');
    const glow = new pc.StandardMaterial();
    glow.emissive = new pc.Color(0.45, 1.0, 0.55);
    glow.emissiveIntensity = 2.2;
    glow.opacity = 0.85;
    glow.blendType = pc.BLEND_ADDITIVE;
    glow.depthWrite = false;
    glow.update();

    this.ring = new pc.Entity('goal-ring');
    this.ring.addComponent('render', { type: 'torus', material: glow });
    this.ring.setLocalScale(0.85, 0.85, 0.85);
    this.root.addChild(this.ring);

    this.beam = new pc.Entity('goal-beam');
    this.beam.addComponent('render', { type: 'cylinder', material: glow });
    this.beam.setLocalScale(0.06, 2.4, 0.06);
    this.beam.setLocalPosition(0, 1.2, 0);
    this.root.addChild(this.beam);

    app.root.addChild(this.root);
  }

  setPosition(x: number, y: number, z: number): void {
    this.root.setPosition(x, y + 0.06, z);
  }

  update(dt: number): void {
    this.t += dt;
    const pulse = 1 + Math.sin(this.t * 3.5) * 0.12;
    this.ring.setLocalScale(0.85 * pulse, 0.85, 0.85 * pulse);
    this.ring.setLocalEulerAngles(0, this.t * 40, 0);
  }
}
