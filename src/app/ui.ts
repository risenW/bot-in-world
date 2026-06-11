// Side control panel + live training charts. Pure DOM, no framework.

import { TrainStats } from '../sim/vectrain';
import { ViewMode } from './level';

export interface UiCallbacks {
  onTrainToggle: () => void;
  onSaveCheckpoint: () => void;
  onLoadCheckpointFile: (file: File) => void;
  onLoadPretrained: () => void;
  onLoadAutosave: () => void;
  onWorldChange: (id: string) => void;
  onLoadCustomWorld: (link: string) => void;
  onNewGoal: () => void;
  onViewMode: (mode: ViewMode) => void;
  onToggleNav: () => void;
  onToggleCameraCollision: () => void;
  onToggleFollow: () => void;
  onToggleGreedy: () => void;
  onRespawn: () => void;
  onSpawnBall: (colorIndex: number) => void;
}

const BALL_COLORS = ['#ef5350', '#42a5f5', '#66bb6a', '#ffee58', '#ab47bc'];

export class Ui {
  private el: HTMLElement;
  private trainBtn!: HTMLButtonElement;
  private stats: Record<string, HTMLElement> = {};
  private chart!: HTMLCanvasElement;
  private returnHistory: number[] = [];
  private successHistory: number[] = [];
  private viewButtons: Record<string, HTMLButtonElement> = {};
  private toggles: Record<string, HTMLButtonElement> = {};
  private worldSelect!: HTMLSelectElement;
  training = false;

  constructor(private cb: UiCallbacks) {
    this.el = document.getElementById('panel')!;
    this.build();
  }

  private build(): void {
    this.el.innerHTML = '';
    this.el.appendChild(h(`
      <div>
        <h1><span class="dot"></span>PufferBot</h1>
        <div class="subtitle">RL navigation in AI-generated worlds — PPO ported from <b>PufferLib</b>, training 100% in your browser.</div>
      </div>`));

    // --- World ---
    const world = group('World');
    const row = div('row');
    this.worldSelect = document.createElement('select');
    this.worldSelect.innerHTML = `
      <option value="warehouse">Obstacle Warehouse (training)</option>
      <option value="gallery">Gallery (unseen world)</option>`;
    this.worldSelect.onchange = () => this.cb.onWorldChange(this.worldSelect.value);
    row.appendChild(this.worldSelect);
    world.appendChild(row);
    const customRow = div('row');
    const linkInput = document.createElement('input');
    linkInput.type = 'text';
    linkInput.placeholder = 'https://app.spaitial.ai/worlds/…';
    linkInput.className = 'link-input';
    const loadBtn = btn('🌍 Load', () => this.cb.onLoadCustomWorld(linkInput.value.trim()));
    linkInput.onkeydown = (e) => { if (e.key === 'Enter') this.cb.onLoadCustomWorld(linkInput.value.trim()); };
    customRow.appendChild(linkInput);
    customRow.appendChild(loadBtn);
    world.appendChild(customRow);
    world.appendChild(h(`<div class="hint">Paste any <b>public</b> Spaitial world link. Swap worlds anytime — the policy only sees egocentric lidar, so what it learned transfers.</div>`));
    this.el.appendChild(world);

    // --- Training ---
    const training = group('Training');
    const trow = div('row');
    this.trainBtn = btn('▶ Start learning', () => this.cb.onTrainToggle());
    this.trainBtn.classList.add('primary');
    trow.appendChild(this.trainBtn);
    training.appendChild(trow);

    const grid = div('stat-grid');
    for (const [key, label] of [
      ['steps', 'Env steps'], ['sps', 'Steps/sec'],
      ['success', 'Success rate'], ['return', 'Mean return'],
      ['eplen', 'Episode len'], ['curriculum', 'Goal dist (max)'],
    ] as const) {
      grid.appendChild(h(`<div class="k">${label}</div>`));
      const v = h(`<div class="v">—</div>`);
      this.stats[key] = v;
      grid.appendChild(v);
    }
    training.appendChild(grid);

    this.chart = document.createElement('canvas');
    this.chart.id = 'chart';
    this.chart.width = 272 * devicePixelRatio;
    this.chart.height = 90 * devicePixelRatio;
    training.appendChild(this.chart);
    training.appendChild(h(`<div class="hint"><span style="color:#4fc3f7">━</span> mean return&nbsp;&nbsp;<span style="color:#69f0ae">━</span> success rate</div>`));
    this.el.appendChild(training);

    // --- Checkpoints ---
    const ckpt = group('Checkpoints');
    const crow1 = div('row');
    crow1.appendChild(btn('💾 Save', () => this.cb.onSaveCheckpoint()));
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.pfbt';
    fileInput.className = 'file-input';
    fileInput.onchange = () => { if (fileInput.files?.[0]) this.cb.onLoadCheckpointFile(fileInput.files[0]); fileInput.value = ''; };
    crow1.appendChild(fileInput);
    crow1.appendChild(btn('📂 Load file', () => fileInput.click()));
    ckpt.appendChild(crow1);
    const crow2 = div('row');
    crow2.appendChild(btn('⚡ Load pretrained', () => this.cb.onLoadPretrained()));
    crow2.appendChild(btn('🕘 Autosave', () => this.cb.onLoadAutosave()));
    ckpt.appendChild(crow2);
    this.el.appendChild(ckpt);

    // --- Bot ---
    const bot = group('Bot');
    const brow = div('row');
    this.toggles.follow = btn('🎥 Follow', () => this.cb.onToggleFollow());
    this.toggles.follow.classList.add('active');
    this.toggles.greedy = btn('🎯 Greedy', () => this.cb.onToggleGreedy());
    brow.appendChild(this.toggles.follow);
    brow.appendChild(this.toggles.greedy);
    brow.appendChild(btn('♻ Respawn <kbd>R</kbd>', () => this.cb.onRespawn()));
    brow.appendChild(btn('🎲 New goal', () => this.cb.onNewGoal()));
    bot.appendChild(brow);
    bot.appendChild(h(`<div class="hint"><b>Click anywhere on the floor</b> to set the goal yourself — the bot walks there. Drag to orbit, wheel to zoom.</div>`));
    this.el.appendChild(bot);

    // --- View ---
    const view = group('View');
    const vrow = div('row');
    const modes: [string, ViewMode, string][] = [
      ['Splat + Bot', 'both', 'B'], ['Mesh overlay', 'mesh-overlay', 'M'],
      ['Splat only', 'splat-only', 'P'], ['Mesh only', 'mesh-only', 'O'],
    ];
    for (const [label, mode, key] of modes) {
      const b = btn(`${label} <kbd>${key}</kbd>`, () => this.cb.onViewMode(mode));
      this.viewButtons[mode] = b;
      vrow.appendChild(b);
    }
    view.appendChild(vrow);
    const vrow2 = div('row');
    this.toggles.nav = btn('Navgrid <kbd>N</kbd>', () => this.cb.onToggleNav());
    this.toggles.cam = btn('Cam collision <kbd>C</kbd>', () => this.cb.onToggleCameraCollision());
    vrow2.appendChild(this.toggles.nav);
    vrow2.appendChild(this.toggles.cam);
    view.appendChild(vrow2);
    this.setViewMode('both');
    this.el.appendChild(view);

    // --- Physics toys ---
    const balls = group('Physics balls');
    const prow = div('row');
    BALL_COLORS.forEach((c, i) => {
      const b = btn(`<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${c}"></span> <kbd>${i + 1}</kbd>`, () => this.cb.onSpawnBall(i));
      prow.appendChild(b);
    });
    balls.appendChild(prow);
    this.el.appendChild(balls);
  }

  setTraining(on: boolean): void {
    this.training = on;
    this.trainBtn.innerHTML = on ? '⏸ Pause learning' : '▶ Start learning';
    this.trainBtn.classList.toggle('active', on);
  }

  setToggle(name: 'follow' | 'greedy' | 'nav' | 'cam', on: boolean): void {
    this.toggles[name]?.classList.toggle('active', on);
  }

  setViewMode(mode: ViewMode): void {
    for (const [m, b] of Object.entries(this.viewButtons)) b.classList.toggle('active', m === mode);
  }

  setWorld(id: string): void {
    this.worldSelect.value = id;
  }

  // register a custom world in the dropdown (or just select it if present)
  addCustomWorld(id: string, label: string): void {
    if (![...this.worldSelect.options].some((o) => o.value === id)) {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = label;
      this.worldSelect.appendChild(opt);
    }
    this.worldSelect.value = id;
  }

  updateStats(s: TrainStats, totalSteps: number): void {
    this.stats.steps.textContent = formatNum(totalSteps);
    this.stats.sps.textContent = formatNum(s.sps);
    this.stats.success.textContent = `${(s.successRate * 100).toFixed(0)} %`;
    this.stats.return.textContent = s.meanReturn.toFixed(2);
    this.stats.eplen.textContent = s.meanEpLen.toFixed(0);
    this.stats.curriculum.textContent = `${s.curriculum.toFixed(1)} m`;
    this.returnHistory.push(s.meanReturn);
    this.successHistory.push(s.successRate);
    if (this.returnHistory.length > 400) { this.returnHistory.shift(); this.successHistory.shift(); }
    this.drawChart();
  }

  private drawChart(): void {
    const ctx = this.chart.getContext('2d')!;
    const W = this.chart.width, H = this.chart.height;
    ctx.clearRect(0, 0, W, H);
    const series = (data: number[], color: string, lo: number, hi: number) => {
      if (data.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 * devicePixelRatio;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * W;
        const yNorm = (data[i] - lo) / Math.max(hi - lo, 1e-6);
        const y = H - 4 - yNorm * (H - 8);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    const rMin = Math.min(...this.returnHistory, 0), rMax = Math.max(...this.returnHistory, 1);
    series(this.returnHistory, '#4fc3f7', rMin, rMax);
    series(this.successHistory, '#69f0ae', 0, 1);
  }

  toast(message: string): void {
    const t = document.getElementById('toast')!;
    t.textContent = message;
    t.classList.add('show');
    clearTimeout((t as any)._timer);
    (t as any)._timer = setTimeout(() => t.classList.remove('show'), 2600);
  }
}

function h(html: string): HTMLElement {
  const tpl = document.createElement('template');
  tpl.innerHTML = html.trim();
  return tpl.content.firstElementChild as HTMLElement;
}
function div(cls: string): HTMLElement {
  const d = document.createElement('div');
  d.className = cls;
  return d;
}
function btn(html: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.innerHTML = html;
  b.onclick = onClick;
  return b;
}
function group(label: string): HTMLElement {
  const g = div('group');
  g.appendChild(h(`<div class="group-label">${label}</div>`));
  return g;
}
function formatNum(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}
