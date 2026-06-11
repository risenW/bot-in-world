# 🐡 PufferBot — watch a bot learn to walk through AI-generated worlds, live in your browser

A humanoid bot teaches itself to navigate **3D worlds generated from a text prompt** ([Spaitial](https://spaitial.ai) Gaussian splats), trained with a TypeScript port of **[PufferLib](https://puffer.ai)'s PPO** — running **100% client-side** in a web worker at thousands of steps per second. No server, no GPU farm: open the page, press *Start learning*, and watch it figure out the world in minutes.

Because the policy only ever sees **egocentric observations** (a 16-ray lidar, the goal bearing in its own body frame, its own speed), it learns *navigation*, not *a map*. Swap in a world it has never seen and it keeps walking.

**Trained 3M steps on the warehouse only:**

| World | Success rate (greedy, 200 episodes) | Avg episode |
| --- | --- | --- |
| Obstacle Warehouse (training world) | 99.5 % | 47 steps |
| Gallery (**never seen during training**) | **100 %** | 44 steps |

## Run it

```bash
npm install
npm run dev        # open the printed URL
```

The repo ships with a pretrained checkpoint (`public/checkpoints/pretrained.pfbt`, 84 KB). The big level assets (`world.ply`, `world.spz`, `mesh_simplified.ply`) are not in git — generate your own worlds (below) or grab them from the project's release assets and drop them into `public/levels/<id>/`.

In the app:

- **▶ Start learning** — PPO training starts in a web worker; the bot in the scene always runs the latest weights, so you literally watch it get smarter. Charts show mean return + success rate.
- **⚡ Load pretrained** — skip to the 3M-step policy.
- **💾 Save / 📂 Load** — checkpoints as `.pfbt` files (puffernet-style flat float32 weights + JSON header). Training also autosaves to localStorage every 15 s.
- **Click anywhere on the floor** to send the bot there.
- **World selector** — swap to the unseen gallery mid-session. Weights carry over; training pauses so you can evaluate first.
- **Load ANY public Spaitial world** — paste a link like `https://app.spaitial.ai/worlds/<id>` into the World panel. The dev server fetches the world's splat, converts it, and the navgrid is baked **from the gaussians alone** (public worlds have no mesh export): floor = the lowest dense band of points per column, obstacles = points at body height. A synthetic ground trimesh keeps click-to-goal and physics working. First load takes a minute; it's cached in `tmp/custom-worlds/` afterwards.
- **Set the goal yourself** — click anywhere on the floor, or hit **🎲 New goal** for a random one.
- View toggles: splat / collision wireframe / navgrid (<kbd>B</kbd> <kbd>M</kbd> <kbd>P</kbd> <kbd>O</kbd> <kbd>N</kbd>), camera collision <kbd>C</kbd>, follow cam <kbd>F</kbd>, greedy policy <kbd>G</kbd>, respawn <kbd>R</kbd>, train toggle <kbd>Space</kbd>, physics balls <kbd>1</kbd>–<kbd>5</kbd>.

## Generate your own worlds

You need a [Spaitial API key](https://developers.spaitial.ai) in `.env` (`SPAITIAL_API_KEY=spt_live_…`).

```bash
# 1. submit a text→world job (5–10 min) and download splat + collision mesh
curl -sX POST https://api.spaitial.ai/v1/worlds \
  -H "Authorization: Bearer $SPAITIAL_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"type":"text","prompt":"an empty cluttered indoor scene… no people"},"validation":{"skip":true}}'
./scripts/fetch-world.sh <request_id> <level-id>

# 2. convert the .spz splat to a PlayCanvas-compatible .ply
npx splat-transform -w public/levels/<level-id>/world.spz public/levels/<level-id>/world.ply

# 3. add public/levels/<level-id>/manifest.json (copy an existing one), then
npm run bake -- --level <level-id>     # navgrid stats + real bounds into the manifest
```

Add the level to the `<select>` in `src/app/ui.ts` and it's playable + trainable. **No retraining required** for a competent bot — that's the point.

## How it works

```
text prompt ──Spaitial API──▶ gaussian splat (.spz→.ply)   visuals (PlayCanvas, unified:false)
                          └─▶ simplified mesh (.ply) ──▶ Rapier trimesh (camera/balls/click-rays)
                                                     └─▶ NavGrid bake: 0.15 m occupancy + height field.
        The reconstructed mesh is a closed "balloon" that extends far beyond the visible world
        (phantom floor planes outside the walls), so floor candidates only count when gaussian
        splat points sit just above them — the splat tells the navgrid where the world is real.
                                                              │
                       ┌──────────────────────────────────────┘
                       ▼
        NavEnv (sim/env.ts): 27-dim egocentric obs ─ 16-ray DDA lidar, goal bearing/distance
        in body frame, speed, collision flag, last action. 6 discrete actions. Reward =
        progress toward goal − step cost − collisions + 5.0 on arrival. Auto-curriculum
        expands goal distance 4 m → 14 m as the success rate passes 80 %.
                       │
                       ▼
        PPO (sim/ppo.ts): TypeScript port of PufferLib's PuffeRL — clipped surrogate +
        clipped value loss, GAE(λ=0.9), grad-norm clip 1.5, Adam(0.95, 0.999), 64 envs ×
        64-step horizon, lr anneal. MLP 27→128→128→{6 logits, value} mirroring puffernet.
        Runs in a web worker (~3–5k steps/s) or in Node (npm run pretrain).
```

Checkpoints are **puffernet-compatible flat weights** (all layers flattened and concatenated in PyTorch order) wrapped with a small JSON header — see `src/sim/checkpoint.ts`.

### Scripts

```bash
npm run pretrain -- --level warehouse --steps 3000000   # headless training → pretrained.pfbt
npx tsx scripts/eval.ts --level gallery --episodes 200  # generalization eval
npm run bake -- --level warehouse                       # navgrid stats / manifest bounds
./scripts/fetch-world.sh <req_id> <level-id>            # download Spaitial artifacts
```

## Credits

- **[PufferLib](https://github.com/PufferAI/PufferLib)** (MIT) — the PPO algorithm, hyperparameter defaults, and the flat-weights checkpoint idea are ported from PuffeRL/puffernet. Go star it.
- **[Spaitial](https://spaitial.ai)** — text-to-3D-world generation (splat + reconstructed mesh).
- **[PlayCanvas](https://playcanvas.com)** — engine + gaussian splat rendering; **[Rapier](https://rapier.rs)** — physics.

## License

MIT — see [LICENSE](LICENSE).
