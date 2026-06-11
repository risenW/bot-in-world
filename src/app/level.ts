// Level loading: splat visual, collision mesh -> Rapier trimesh + NavGrid,
// debug overlays (collision wireframe, navgrid cells), view modes.

import * as pc from 'playcanvas';
import RAPIER from '@dimforge/rapier3d-compat';
import { parsePly, ParsedMesh } from '../sim/ply';
import { bakeNavGrid, bakeNavGridFromSplat, gridToMesh, NavGrid, cellCenter } from '../sim/navgrid';

export interface LevelManifest {
  id: string;
  name: string;
  description: string;
  splats: { url: string }[];
  collision: { url: string }[];
}

export type ViewMode = 'both' | 'mesh-overlay' | 'splat-only' | 'mesh-only';

export class Level {
  app: pc.Application;
  world: RAPIER.World;
  manifest!: LevelManifest;
  mesh!: ParsedMesh;
  grid!: NavGrid;
  levelCollider!: RAPIER.Collider;

  splatEntity: pc.Entity | null = null;
  private splatAsset: pc.Asset | null = null;
  private splatPts: Float32Array | null = null;
  climbHeight = 0.5;   // max climbable step; rebake() applies changes
  debugMesh: pc.Entity | null = null;
  navDebug: pc.Entity | null = null;
  mode: ViewMode = 'both';
  navVisible = false;

  constructor(app: pc.Application, world: RAPIER.World) {
    this.app = app;
    this.world = world;
  }

  // levelId: a bundled level ("warehouse"), "created:<request_id>" for a world
  // generated through the BYOK dialog (full splat + mesh pipeline), or
  // "custom:<world-uuid>" for a public link (splat only, no mesh).
  async load(levelId: string, onProgress: (step: string, frac: number) => void): Promise<void> {
    this.unload();
    const customUuid = levelId.startsWith('custom:') ? levelId.slice(7) : null;
    const createdId = levelId.startsWith('created:') ? levelId.slice(8) : null;

    if (customUuid) {
      this.manifest = {
        id: levelId,
        name: `Spaitial ${customUuid.slice(0, 8)}`,
        description: 'Loaded from a public Spaitial link',
        splats: [{ url: `/ext/spaitial/${customUuid}/world.ply` }],
        collision: [],
      };
      onProgress('Fetching + converting world — first load can take a couple of minutes…', 0.1);
      await this.loadSplat(this.manifest.splats[0].url);

      const points = this.splatPositions();
      if (!points) throw new Error('could not read gaussian data from the loaded splat');
      this.splatPts = points;
      onProgress('Baking navigation grid from splat…', 0.6);
      await microtask();
      // no mesh export for public worlds: navgrid from the splat itself,
      // plus a synthetic ground trimesh so raycasts and physics still work
      this.grid = bakeNavGridFromSplat(points);
      const gm = gridToMesh(this.grid);
      this.mesh = {
        positions: gm.positions,
        indices: gm.indices,
        aabb: { min: [this.grid.minX, this.grid.meshMinY, this.grid.minZ], max: [this.grid.minX + this.grid.w * this.grid.cell, this.grid.meshMinY + 10, this.grid.minZ + this.grid.h * this.grid.cell] },
      };
    } else {
      if (createdId) {
        let name = `Created ${createdId.slice(4, 10)}`;
        try {
          const st = await (await fetch(`/ext/created/${createdId}/status`)).json();
          if (st.title) name = st.title;
        } catch { /* name fallback is fine */ }
        this.manifest = {
          id: levelId,
          name,
          description: 'Generated with your Spaitial API key',
          splats: [{ url: `/ext/created/${createdId}/world.ply` }],
          collision: [{ url: `/ext/created/${createdId}/mesh_simplified.ply` }],
        };
      } else {
        onProgress('Loading manifest…', 0.05);
        this.manifest = (await (await fetch(`/levels/${levelId}/manifest.json`)).json()) as LevelManifest;
      }
      const manifest = this.manifest;

      onProgress('Loading gaussian splat…', 0.1);
      await this.loadSplat(manifest.splats[0].url);
      this.splatPts = this.splatPositions();

      onProgress('Downloading collision mesh…', 0.45);
      const meshBuf = await (await fetch(manifest.collision[0].url)).arrayBuffer();
      onProgress('Parsing collision mesh…', 0.6);
      this.mesh = parsePly(meshBuf);

      onProgress('Baking navigation grid…', 0.7);
      await microtask();
      this.grid = bakeNavGrid(this.mesh, this.splatPts ?? undefined, { climbHeight: this.climbHeight });
    }
    if (this.grid.spawn.length < 50) {
      console.warn(`navgrid: only ${this.grid.spawn.length} walkable cells — world may be hard to navigate`);
    }

    onProgress('Creating physics…', 0.85);
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    this.levelCollider = this.world.createCollider(
      RAPIER.ColliderDesc.trimesh(this.mesh.positions, this.mesh.indices),
      body,
    );

    onProgress('Building debug views…', 0.92);
    this.buildDebugMesh();
    this.buildNavDebug();
    this.applyMode();
    onProgress('Ready', 1);
  }

  // re-bake the navgrid with a new climb height (mesh-based levels only)
  rebake(climbHeight: number): void {
    this.climbHeight = climbHeight;
    if (this.manifest.collision.length > 0) {
      this.grid = bakeNavGrid(this.mesh, this.splatPts ?? undefined, { climbHeight });
      this.navDebug?.destroy();
      this.buildNavDebug();
      if (this.navDebug) (this.navDebug as pc.Entity).enabled = this.navVisible;
    }
  }

  unload(): void {
    this.splatEntity?.destroy(); this.splatEntity = null;
    this.splatPts = null;
    if (this.splatAsset) {
      this.app.assets.remove(this.splatAsset);
      this.splatAsset.unload();
      this.splatAsset = null;
    }
    this.debugMesh?.destroy(); this.debugMesh = null;
    this.navDebug?.destroy(); this.navDebug = null;
    if (this.levelCollider) {
      const body = this.levelCollider.parent();
      this.world.removeCollider(this.levelCollider, false);
      if (body) this.world.removeRigidBody(body);
      this.levelCollider = undefined as never;
    }
  }

  // Pull gaussian center positions out of the loaded gsplat resource for the
  // navgrid's spawn-density filter. Engine-internal but guarded; returns null
  // if the engine layout changes (spawning then falls back to the full region).
  private splatPositions(): Float32Array | null {
    try {
      const res = this.splatAsset?.resource as any;
      const data = res?.gsplatData ?? res?.splatData ?? null;
      if (!data?.getProp) return null;
      const x = data.getProp('x') as Float32Array | undefined;
      const y = data.getProp('y') as Float32Array | undefined;
      const z = data.getProp('z') as Float32Array | undefined;
      if (!x || !y || !z) return null;
      const out = new Float32Array(x.length * 3);
      for (let i = 0; i < x.length; i++) {
        out[i * 3] = x[i]; out[i * 3 + 1] = y[i]; out[i * 3 + 2] = z[i];
      }
      return out;
    } catch (e) {
      console.warn('splatPositions failed', e);
      return null;
    }
  }

  private loadSplat(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const asset = new pc.Asset(`splat-${this.manifest.id}`, 'gsplat' as never, { url });
      this.splatAsset = asset;
      asset.once('load', () => {
        const entity = new pc.Entity('world-splat');
        (entity as any).addComponent('gsplat', { asset, unified: false });
        this.app.root.addChild(entity);
        this.splatEntity = entity;
        resolve();
      });
      asset.once('error', (err: string) => reject(new Error(err)));
      this.app.assets.add(asset);
      this.app.assets.load(asset);
    });
  }

  private buildDebugMesh(): void {
    const { positions, indices } = this.mesh;
    const linePositions = new Float32Array(indices.length * 2 * 3);
    let o = 0;
    for (let t = 0; t < indices.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const a = indices[t + e] * 3;
        const b = indices[t + ((e + 1) % 3)] * 3;
        linePositions[o++] = positions[a]; linePositions[o++] = positions[a + 1]; linePositions[o++] = positions[a + 2];
        linePositions[o++] = positions[b]; linePositions[o++] = positions[b + 1]; linePositions[o++] = positions[b + 2];
      }
    }
    const mesh = new pc.Mesh(this.app.graphicsDevice);
    mesh.setPositions(linePositions);
    mesh.update(pc.PRIMITIVE_LINES, true);

    const material = new pc.StandardMaterial();
    material.diffuse = new pc.Color(0.1, 0.85, 1.0);
    material.emissive = new pc.Color(0.08, 0.65, 0.9);
    material.opacity = 0.72;
    material.blendType = pc.BLEND_NORMAL;
    material.cull = pc.CULLFACE_NONE;
    material.depthWrite = false;
    material.update();

    const entity = new pc.Entity('collision-debug');
    entity.addComponent('render', { meshInstances: [new pc.MeshInstance(mesh, material)] });
    this.app.root.addChild(entity);
    this.debugMesh = entity;
  }

  private buildNavDebug(): void {
    // Walkable cells as a point grid slightly above ground.
    const g = this.grid;
    const stride = Math.max(1, Math.floor(g.spawn.length / 30000));
    const count = Math.ceil(g.spawn.length / stride);
    const positions = new Float32Array(count * 3);
    let o = 0;
    for (let i = 0; i < g.spawn.length; i += stride) {
      const ci = g.spawn[i];
      const [x, z] = cellCenter(g, ci);
      positions[o++] = x;
      positions[o++] = g.ground[ci] + 0.04;
      positions[o++] = z;
    }
    const mesh = new pc.Mesh(this.app.graphicsDevice);
    mesh.setPositions(positions);
    mesh.update(pc.PRIMITIVE_POINTS, true);

    const material = new pc.StandardMaterial();
    material.emissive = new pc.Color(0.3, 1.0, 0.5);
    material.update();

    const entity = new pc.Entity('nav-debug');
    entity.addComponent('render', { meshInstances: [new pc.MeshInstance(mesh, material)] });
    entity.enabled = false;
    this.app.root.addChild(entity);
    this.navDebug = entity;
  }

  setMode(mode: ViewMode): void {
    this.mode = mode;
    this.applyMode();
  }

  toggleNav(): boolean {
    this.navVisible = !this.navVisible;
    if (this.navDebug) this.navDebug.enabled = this.navVisible;
    return this.navVisible;
  }

  private applyMode(): void {
    const splat = this.splatEntity, mesh = this.debugMesh;
    if (splat) splat.enabled = this.mode !== 'mesh-only';
    if (mesh) mesh.enabled = this.mode === 'mesh-overlay' || this.mode === 'mesh-only';
  }
}

function microtask(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}
