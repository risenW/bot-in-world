# 🤖 SpAItial Bot — watch a bot learn to walk through AI-generated worlds, live in your browser

A humanoid bot teaches itself to navigate **3D worlds generated from a text prompt or a photo** with the [Spaitial API](https://developers.spaitial.ai) (gaussian splat + reconstructed collision mesh), trained with a TypeScript port of **[PufferLib](https://puffer.ai)'s PPO**. The reinforcement learning runs **100% client-side** in a web worker at thousands of steps per second — no training server, no GPU farm. Open the page, press *Start learning*, and watch it figure out the world in minutes.

**▶ [Live demo](https://risenw.github.io/spaitial-bot/)** — two bundled worlds, train from scratch or load the pretrained policy, all in your browser. (Creating *new* worlds from the Spaitial API needs a quick local setup — see below.)

> The small Vite dev server is only a thin proxy for **world generation** — it forwards Spaitial API calls and converts the generated `.spz` splat to a PlayCanvas-loadable splat. All RL training and inference is pure browser, which is why the hosted demo works as a static site.

Bring your own Spaitial API key, type a prompt (or drop in a photo), and a few minutes later the bot is learning to walk, climb, and fetch in *your* world.

Because the policy only ever sees **egocentric observations** (a 16-ray lidar, the goal bearing in its own body frame, its own speed), it learns *navigation*, not *a map*. Swap in a world it has never seen and it keeps walking.

**Trained on the gallery only, evaluated greedy:**

| Task | World | Success rate | Avg episode |
| --- | --- | --- | --- |
| Navigate (3M steps) | Gallery (training world) | 100 % | 43 steps |
| Navigate | Obstacle Warehouse (**never seen during training**) | **96.5 %** | 64 steps |
| Fetch 4 balls (8M steps) | Gallery (training world) | 100 % | 445 steps |
| Fetch 4 balls | Obstacle Warehouse (**never seen**) | **82.0 %** | 620 steps |

## Run it

```bash
npm install
npm run dev        # open the printed URL
```

The repo ships ready to play: two pretrained checkpoints (`pretrained.pfbt` / `pretrained-fetch.pfbt`, ~85 KB each) and two worlds as compressed `.sog` splats + collision meshes (~30 MB each) — the same assets the [live demo](https://risenw.github.io/spaitial-bot/) runs on. The uncompressed splat sources (`world.ply`, `world.spz`) are gitignored; you only need them to re-run the offline scripts.

### Hosted demo vs. local

Everything except world creation is pure static frontend, so the [live demo](https://risenw.github.io/spaitial-bot/) (auto-deployed to GitHub Pages) lets anyone train and watch the bot in the bundled worlds. **Creating or importing worlds via the Spaitial API needs the local dev server**  for the `.spz`→splat conversion, which is a Node step, so it can't run on a static host. Clone + `npm run dev` and the "Create your own world" button lights up.

In the app:

- **Pretrained on arrival** — every scene loads the task's pretrained policy automatically, so the bot is competent the moment a world appears (and it carries that policy into unseen worlds — the generalization demo).
- **▶ Start learning** — PPO training starts in a web worker and **fine-tunes** from the current policy; the bot in the scene always runs the latest weights, so you watch it improve live. Charts show mean return + success rate.
- **↺ Reset** — wipe to a random policy to watch the bot learn from scratch.
- **⚡ Load pretrained** — re-load the bundled policy for the current task (3M-step navigate / 8M-step fetch).
- **💾 Save / 📂 Load** — checkpoints as `.pfbt` files (puffernet-style flat float32 weights + JSON header). Training also autosaves to localStorage every 15 s.
- **Click anywhere on the floor** to send the bot there.
- **World selector** — swap to the unseen gallery mid-session. Weights carry over; training pauses so you can evaluate first.
- **✨ Create your own world (BYOK)** — open the dialog, paste your [Spaitial API key](https://developers.spaitial.ai), and describe a world (or upload a photo). The local server submits the generation, polls it, downloads the splat **and the reconstructed collision mesh**, converts everything, and drops the bot in — full physics, climbing, and training support. Your key stays in memory on your machine and is never written to disk. Generation takes ~5–10 minutes, but you don't have to wait around: a **live status chip** in the panel shows the current phase and elapsed time, you can close the dialog or keep using the app, and it **survives a page reload** (the job is re-attached to the still-running Spaitial request). When it's done the world appears in the World list.
- **♻️ Reuse an existing world** — already generated a world with your key? In the same dialog, paste its **request ID** (`req_…`, found at [developers.spaitial.ai/generations](https://developers.spaitial.ai/generations)) and Import. It pulls that world's splat **and mesh** straight from the API — no regeneration.
- **Climbing** — the bot hops up onto low platforms, crates, and steps like a tiny parkour humanoid. Set the max climb height (off / 0.35 / 0.5 / 0.7 m) in the Bot panel; the navgrid re-bakes climbable tops on the fly (low flat surfaces get promoted to walkable ground, crate rims are bridged as climb seams).
- **Set the goal yourself** — click anywhere on the floor, or hit **🎲 New goal** for a random one.
- **Two tasks** — 🧭 *Navigate* (walk to the goal) and 🔵 *Fetch balls*: a configurable number of blue balls are scattered through the world; the bot must find each one, pick it up by touching it, and carry it back to the green goal ring — fast, without hitting obstacles. Other-colored balls are decoys it never senses. Training auto-curriculums from 1 ball up to your setting. The two tasks share one network (32-dim egocentric obs), so navigation skill transfers into fetch training.
- **Camera** — drag to orbit, wheel to zoom, <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to fly and <kbd>Q</kbd>/<kbd>E</kbd> down/up (turns Follow off); re-enabling Follow snaps back behind the bot.
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
        NavEnv (sim/env.ts): 32-dim egocentric obs ─ 16-ray DDA lidar, goal bearing/distance
        in body frame, speed, collision flag, last action, plus fetch channels (carrying,
        nearest target-ball bearing/distance, fraction remaining). 6 discrete actions.
        Reward = progress toward the current objective − step cost − collisions, + pickup /
        deposit / completion bonuses. Auto-curriculum grows goal distance (4→14 m) and, in
        fetch, ball count (1→N) as the success rate climbs. One obs layout for both tasks,
        so a nav policy transfers straight into fetch training.
                       │
                       ▼
        PPO (sim/ppo.ts): TypeScript port of PufferLib's PuffeRL — clipped surrogate +
        clipped value loss, GAE(λ=0.9), grad-norm clip 1.5, Adam(0.95, 0.999), 64 envs ×
        64-step horizon, lr anneal. MLP 32→128→128→{6 logits, value} mirroring puffernet.
        Runs in a web worker (~3–5k steps/s) or in Node (npm run pretrain).
```

Checkpoints are **puffernet-compatible flat weights** (all layers flattened and concatenated in PyTorch order) wrapped with a small JSON header — see `src/sim/checkpoint.ts`.

### Scripts

```bash
npm run pretrain -- --level gallery --steps 3000000              # nav → pretrained.pfbt
npm run pretrain -- --level gallery --task fetch --steps 8000000 # fetch → pretrained-fetch.pfbt
npx tsx scripts/eval.ts --level warehouse --episodes 200         # generalization eval (unseen)
npx tsx scripts/eval.ts --level warehouse --task fetch           # fetch eval on unseen world
npm run bake -- --level gallery                                  # navgrid stats / manifest bounds
./scripts/fetch-world.sh <req_id> <level-id>                     # download Spaitial artifacts
```

## Credits

- **[PufferLib](https://github.com/PufferAI/PufferLib)** (MIT) — the PPO algorithm, hyperparameter defaults, and the flat-weights checkpoint idea are ported from PuffeRL/puffernet. Go star it.
- **[Spaitial](https://spaitial.ai)** — text-to-3D-world generation (splat + reconstructed mesh).
- **[PlayCanvas](https://playcanvas.com)** — engine + gaussian splat rendering; **[Rapier](https://rapier.rs)** — physics.

## License

MIT — see [LICENSE](LICENSE).
