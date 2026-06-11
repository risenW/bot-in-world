import { defineConfig, type Plugin } from 'vite';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { existsSync, mkdirSync, createReadStream, statSync, writeFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { Readable } from 'node:stream';

// Serves arbitrary PUBLIC Spaitial worlds to the app:
//   GET /ext/spaitial/:uuid/meta       -> proxied signed-urls JSON (their API has no CORS for us)
//   GET /ext/spaitial/:uuid/world.ply  -> downloads the .spz, converts via splat-transform, caches on disk
// First conversion takes a minute or two; afterwards it's served straight from tmp/custom-worlds/.

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const CACHE_DIR = resolve(__dirname, 'tmp/custom-worlds');
const inflight = new Map<string, Promise<string>>();

async function preparePly(uuid: string): Promise<string> {
  const dir = resolve(CACHE_DIR, uuid);
  const ply = resolve(dir, 'world.ply');
  if (existsSync(ply)) return ply;
  if (inflight.has(uuid)) return inflight.get(uuid)!;

  const job = (async () => {
    mkdirSync(dir, { recursive: true });
    const metaRes = await fetch(`https://api.spaitial.ai/worlds/public/${uuid}/signed-urls`);
    if (!metaRes.ok) throw new Error(`signed-urls ${metaRes.status} — is the world public?`);
    const meta = (await metaRes.json()) as { splat_url?: string };
    if (!meta.splat_url) throw new Error('world has no splat_url');
    writeFileSync(resolve(dir, 'meta.json'), JSON.stringify(meta, null, 2));

    const spz = resolve(dir, 'world.spz');
    if (!existsSync(spz)) {
      const dl = await fetch(meta.splat_url);
      if (!dl.ok || !dl.body) throw new Error(`splat download ${dl.status}`);
      const { writeFile } = await import('node:fs/promises');
      await writeFile(spz, Readable.fromWeb(dl.body as never));
    }

    await new Promise<void>((res, rej) => {
      execFile('npx', ['splat-transform', '-w', spz, ply], { cwd: __dirname, timeout: 10 * 60 * 1000 },
        (err, _out, stderr) => (err ? rej(new Error(`splat-transform failed: ${stderr?.slice(-400)}`)) : res()));
    });
    return ply;
  })();
  inflight.set(uuid, job);
  try {
    return await job;
  } finally {
    inflight.delete(uuid);
  }
}

function handler() {
  return async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    const m = req.url?.match(/^\/ext\/spaitial\/([0-9a-f-]{36})\/(meta|world\.ply)$/);
    if (!m) return next();
    const [, uuid, what] = m;
    if (!UUID_RE.test(uuid)) { res.statusCode = 400; res.end('bad uuid'); return; }
    try {
      if (what === 'meta') {
        const r = await fetch(`https://api.spaitial.ai/worlds/public/${uuid}/signed-urls`);
        res.statusCode = r.status;
        res.setHeader('content-type', 'application/json');
        res.end(await r.text());
        return;
      }
      const ply = await preparePly(uuid);
      const size = statSync(ply).size;
      res.setHeader('content-type', 'application/octet-stream');
      res.setHeader('content-length', String(size));
      createReadStream(ply).pipe(res);
    } catch (e) {
      res.statusCode = 502;
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ error: (e as Error).message }));
    }
  };
}

function spaitialProxy(): Plugin {
  return {
    name: 'spaitial-public-worlds',
    configureServer(server) { server.middlewares.use(handler()); },
    configurePreviewServer(server) { server.middlewares.use(handler()); },
  };
}

export default defineConfig({
  plugins: [spaitialProxy()],
});
