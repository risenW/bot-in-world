import { defineConfig, type Plugin } from 'vite';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { existsSync, mkdirSync, createReadStream, statSync, writeFileSync, readFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { Readable } from 'node:stream';

// Server-side companion for the app (dev + preview):
//
// BYOK world creation (the main path — full splat + collision mesh):
//   POST /ext/create-world                    body {prompt?|imageBase64?, title?}, header x-spaitial-key
//   GET  /ext/created/:reqId/status           {phase: generating|downloading|mesh-export|converting|ready|error}
//   GET  /ext/created/:reqId/world.ply        converted splat
//   GET  /ext/created/:reqId/mesh_simplified.ply
//
// Public-link worlds (splat only, no mesh export available):
//   GET  /ext/spaitial/:uuid/meta             proxied signed-urls JSON
//   GET  /ext/spaitial/:uuid/world.ply        downloaded + converted, disk-cached
//
// The user's API key is held in memory for the lifetime of a creation job and
// never written to disk or logged.

const API = 'https://api.spaitial.ai';
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const REQ_RE = /^req_[0-9a-f]{32}$/;
const PUBLIC_CACHE = resolve(__dirname, 'tmp/custom-worlds');
const CREATED_CACHE = resolve(__dirname, 'tmp/created-worlds');
const NO_PEOPLE = ' No people, no humans, no characters, no crowds, no portraits, no mannequins, no animals, no visible body parts.';

const inflight = new Map<string, Promise<string>>();

interface CreateJob { phase: string; detail?: string; error?: string; title?: string }
const jobs = new Map<string, CreateJob>();

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function convertSpz(spz: string, ply: string): Promise<void> {
  return new Promise((res, rej) => {
    execFile('npx', ['splat-transform', '-w', spz, ply], { cwd: __dirname, timeout: 10 * 60 * 1000 },
      (err, _out, stderr) => (err ? rej(new Error(`splat-transform failed: ${stderr?.slice(-400)}`)) : res()));
  });
}

async function downloadTo(url: string, headers: Record<string, string>, dest: string): Promise<void> {
  const r = await fetch(url, { headers });
  if (!r.ok || !r.body) throw new Error(`download ${r.status}`);
  await writeFile(dest, Readable.fromWeb(r.body as never));
}

async function readBody(req: IncomingMessage, limit = 40 * 1024 * 1024): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > limit) throw new Error('body too large');
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function json(res: ServerResponse, status: number, data: unknown): void {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(data));
}

function serveFile(res: ServerResponse, path: string): void {
  if (!existsSync(path)) { res.statusCode = 404; res.end('not ready'); return; }
  res.setHeader('content-type', 'application/octet-stream');
  res.setHeader('content-length', String(statSync(path).size));
  createReadStream(path).pipe(res);
}

// ---------------- BYOK creation ----------------

async function runCreateJob(reqId: string, key: string): Promise<void> {
  const job = jobs.get(reqId)!;
  const dir = resolve(CREATED_CACHE, reqId);
  mkdirSync(dir, { recursive: true });
  const auth = { Authorization: `Bearer ${key}` };
  try {
    job.phase = 'generating';
    for (;;) {
      const st = (await (await fetch(`${API}/v1/worlds/requests/${reqId}/status`, { headers: auth })).json()) as { status: string; progress?: number };
      if (st.status === 'COMPLETED') break;
      if (st.status === 'FAILED' || st.status === 'CANCELLED') throw new Error(`generation ${st.status}`);
      job.detail = st.progress != null ? `${Math.round(st.progress * 100)}%` : undefined;
      await sleep(8000);
    }
    try {
      const env = (await (await fetch(`${API}/v1/worlds/requests/${reqId}`, { headers: auth })).json()) as { world?: { title?: string } };
      if (env.world?.title) job.title = env.world.title;
    } catch { /* title is cosmetic */ }

    job.phase = 'downloading';
    job.detail = undefined;
    const spz = resolve(dir, 'world.spz');
    if (!existsSync(spz)) await downloadTo(`${API}/v1/worlds/requests/${reqId}/splat`, auth, spz);

    job.phase = 'mesh-export';
    await fetch(`${API}/v1/worlds/requests/${reqId}/exports/mesh-simplified`, { method: 'POST', headers: auth });
    for (;;) {
      const ex = (await (await fetch(`${API}/v1/worlds/requests/${reqId}/exports/mesh-simplified`, { headers: auth })).json()) as { status: string; download_url?: string };
      if (ex.status === 'READY' && ex.download_url) {
        await downloadTo(ex.download_url, auth, resolve(dir, 'mesh_simplified.ply'));
        break;
      }
      if (ex.status === 'FAILED') throw new Error('mesh export failed');
      await sleep(8000);
    }

    job.phase = 'converting';
    await convertSpz(spz, resolve(dir, 'world.ply'));
    writeFileSync(resolve(dir, 'meta.json'), JSON.stringify({ request_id: reqId, title: job.title ?? null }, null, 2));
    job.phase = 'ready';
  } catch (e) {
    job.phase = 'error';
    job.error = (e as Error).message;
  }
}

// Resolve a user-pasted id to a request_id we can run the mesh pipeline on.
// Accepts a req_… directly, or a world UUID / app.spaitial.ai link (resolved
// by scanning this key's recent requests for a matching world.id).
async function resolveRequestId(raw: string, key: string): Promise<string> {
  const reqMatch = raw.match(/req_[0-9a-f]{32}/);
  if (reqMatch) return reqMatch[0];
  const uuidMatch = raw.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i);
  if (!uuidMatch) throw new Error('paste a request ID (req_…) or a world link / ID');
  const uuid = uuidMatch[0].toLowerCase();
  const auth = { Authorization: `Bearer ${key}` };
  const PAGE = 40, MAX_SCAN = 120;
  let offset = 0, scanned = 0;
  while (scanned < MAX_SCAN) {
    const listRes = await fetch(`${API}/v1/worlds/requests?limit=${PAGE}&offset=${offset}`, { headers: auth });
    if (!listRes.ok) throw new Error(`could not list your worlds (${listRes.status})`);
    const list = (await listRes.json()) as { requests?: { request_id: string }[]; has_more?: boolean };
    const reqs = list.requests ?? [];
    if (reqs.length === 0) break;
    const details = await Promise.all(reqs.map((r) =>
      fetch(`${API}/v1/worlds/requests/${r.request_id}`, { headers: auth }).then((x) => x.json()).catch(() => null)));
    for (const d of details) {
      const w = (d as { world?: { id?: string } } | null)?.world;
      if (w?.id?.toLowerCase() === uuid) return (d as { request_id: string }).request_id;
    }
    scanned += reqs.length;
    offset += PAGE;
    if (!list.has_more) break;
  }
  throw new Error('that world is not among your API key’s requests — paste its request ID (req_…) instead');
}

function createdStatus(reqId: string): CreateJob | null {
  const job = jobs.get(reqId);
  if (job) return job;
  const dir = resolve(CREATED_CACHE, reqId);
  if (existsSync(resolve(dir, 'world.ply')) && existsSync(resolve(dir, 'mesh_simplified.ply'))) {
    let title: string | undefined;
    try { title = JSON.parse(readFileSync(resolve(dir, 'meta.json'), 'utf8')).title ?? undefined; } catch { /* ok */ }
    return { phase: 'ready', title };
  }
  return null;
}

// ---------------- public-link worlds (splat only) ----------------

async function preparePublicPly(uuid: string): Promise<string> {
  const dir = resolve(PUBLIC_CACHE, uuid);
  const ply = resolve(dir, 'world.ply');
  if (existsSync(ply)) return ply;
  if (inflight.has(uuid)) return inflight.get(uuid)!;
  const work = (async () => {
    mkdirSync(dir, { recursive: true });
    const metaRes = await fetch(`${API}/worlds/public/${uuid}/signed-urls`);
    if (!metaRes.ok) throw new Error(`signed-urls ${metaRes.status} — is the world public?`);
    const meta = (await metaRes.json()) as { splat_url?: string };
    if (!meta.splat_url) throw new Error('world has no splat_url');
    const spz = resolve(dir, 'world.spz');
    if (!existsSync(spz)) await downloadTo(meta.splat_url, {}, spz);
    await convertSpz(spz, ply);
    return ply;
  })();
  inflight.set(uuid, work);
  try { return await work; } finally { inflight.delete(uuid); }
}

// ---------------- middleware ----------------

function handler() {
  return async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    const url = req.url ?? '';
    try {
      if (url === '/ext/create-world' && req.method === 'POST') {
        const key = (req.headers['x-spaitial-key'] as string | undefined)?.trim();
        if (!key || !key.startsWith('spt_')) { json(res, 401, { error: 'missing or invalid x-spaitial-key' }); return; }
        const body = JSON.parse(await readBody(req)) as { prompt?: string; imageBase64?: string; title?: string };
        let input: Record<string, unknown>;
        if (body.prompt?.trim()) input = { type: 'text', prompt: body.prompt.trim() + NO_PEOPLE };
        else if (body.imageBase64) input = { type: 'base64', image_base64: body.imageBase64 };
        else { json(res, 400, { error: 'provide a prompt or an image' }); return; }

        const submit = await fetch(`${API}/v1/worlds`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input,
            title: body.title || undefined,
            validation: { skip: true },
            visibility: { is_public: false, is_listed: false },
          }),
        });
        const submitText = await submit.text();
        if (!submit.ok) { res.statusCode = submit.status; res.setHeader('content-type', 'application/json'); res.end(submitText); return; }
        const { request_id } = JSON.parse(submitText) as { request_id: string };
        jobs.set(request_id, { phase: 'generating', title: body.title });
        void runCreateJob(request_id, key);
        json(res, 202, { request_id });
        return;
      }

      // import / reuse a world already generated with this key (by request_id
      // or world UUID). Runs the same download + mesh-export + convert pipeline.
      if (url === '/ext/import-world' && req.method === 'POST') {
        const key = (req.headers['x-spaitial-key'] as string | undefined)?.trim();
        if (!key || !key.startsWith('spt_')) { json(res, 401, { error: 'missing or invalid x-spaitial-key' }); return; }
        const body = JSON.parse(await readBody(req, 4096)) as { id?: string };
        const reqId = await resolveRequestId(String(body.id ?? '').trim(), key);
        const st = createdStatus(reqId);
        if (!st || st.phase === 'error') { jobs.set(reqId, { phase: 'generating' }); void runCreateJob(reqId, key); }
        json(res, 202, { request_id: reqId });
        return;
      }

      // re-attach a job after a dev-server restart (jobs are in-memory). The
      // client still has the key; the world keeps generating on Spaitial's side.
      const resume = url.match(/^\/ext\/created\/(req_[0-9a-f]{32})\/resume$/);
      if (resume && req.method === 'POST') {
        const reqId = resume[1];
        if (!REQ_RE.test(reqId)) { res.statusCode = 400; res.end('bad id'); return; }
        const existing = createdStatus(reqId);
        if (existing) { json(res, 200, existing); return; }
        const key = (req.headers['x-spaitial-key'] as string | undefined)?.trim();
        if (!key || !key.startsWith('spt_')) { json(res, 409, { error: 'need key to resume' }); return; }
        jobs.set(reqId, { phase: 'generating' });
        void runCreateJob(reqId, key);
        json(res, 202, { phase: 'generating' });
        return;
      }

      const created = url.match(/^\/ext\/created\/(req_[0-9a-f]{32})\/(status|world\.ply|mesh_simplified\.ply)$/);
      if (created) {
        const [, reqId, what] = created;
        if (!REQ_RE.test(reqId)) { res.statusCode = 400; res.end('bad id'); return; }
        if (what === 'status') {
          const st = createdStatus(reqId);
          if (!st) json(res, 404, { error: 'unknown world' });
          else json(res, 200, st);
        } else {
          serveFile(res, resolve(CREATED_CACHE, reqId, what));
        }
        return;
      }

      const pub = url.match(/^\/ext\/spaitial\/([0-9a-f-]{36})\/(meta|world\.ply)$/);
      if (pub) {
        const [, uuid, what] = pub;
        if (!UUID_RE.test(uuid)) { res.statusCode = 400; res.end('bad uuid'); return; }
        if (what === 'meta') {
          const r = await fetch(`${API}/worlds/public/${uuid}/signed-urls`);
          res.statusCode = r.status;
          res.setHeader('content-type', 'application/json');
          res.end(await r.text());
        } else {
          serveFile(res, await preparePublicPly(uuid));
        }
        return;
      }
    } catch (e) {
      json(res, 502, { error: (e as Error).message });
      return;
    }
    next();
  };
}

function spaitialPlugin(): Plugin {
  return {
    name: 'spaitial-worlds',
    configureServer(server) { server.middlewares.use(handler()); },
    configurePreviewServer(server) { server.middlewares.use(handler()); },
  };
}

export default defineConfig({
  plugins: [spaitialPlugin()],
});
