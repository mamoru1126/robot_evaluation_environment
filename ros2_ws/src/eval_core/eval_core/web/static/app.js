/* Robot evaluation environment - browser UI (vanilla JS + Plotly) */
'use strict';

const S = {
  info: null,
  controllers: [],
  ctrlErrors: [],
  presets: [],
  scenario: null,
  preview: null,
  ctrlState: {},          // name -> {checked, params}
  live: null,
  trails: { id: null, robot: [], humans: [] },
  batches: [],
  selection: new Set(),
  detailRef: null,
  analysis: null,
};
const HUMAN_COLORS = ['#e6550d', '#3182bd', '#31a354', '#756bb1'];
const CTRL_COLORS = { pure_pursuit: '#1f77b4', dwpp: '#ff7f0e', dwa: '#2ca02c', mppi: '#d62728', mpc: '#9467bd' };
const EXTRA_COLORS = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
const OUTCOME_LABEL = { goal: 'GOAL', collision: '接触', timeout: 'タイムアウト', aborted: '中断', error: 'エラー' };
const STATE_LABEL = {
  stopping: '停止中', resetting: 'リセット', settling: '待機', starting_controller: '制御器起動',
  running: '走行中', finished: '完了',
};

// ------------------------------------------------------------------ utils
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}
async function api(path, opts = {}) {
  const init = { method: opts.method || 'GET', headers: {} };
  if (opts.body !== undefined) {
    init.method = opts.method || 'POST';
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.body);
  }
  const r = await fetch(path, init);
  if (opts.raw) {
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r;
  }
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  if (!r.ok) throw new Error((data && (data.error || data.message)) || `HTTP ${r.status}`);
  return data;
}
let toastTimer = null;
function toast(msg, isError = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 7000 : 3000);
}
function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  if (typeof v === 'number') return v.toFixed(d);
  return String(v);
}
function ctrlColor(name, order = []) {
  if (CTRL_COLORS[name]) return CTRL_COLORS[name];
  const extras = order.filter(n => !CTRL_COLORS[n]);
  const i = Math.max(0, extras.indexOf(name));
  return EXTRA_COLORS[i % EXTRA_COLORS.length];
}
function getPath(obj, path) { return path.split('.').reduce((o, k) => (o ? o[k] : undefined), obj); }
function setPath(obj, path, value) {
  const ks = path.split('.');
  let o = obj;
  for (const k of ks.slice(0, -1)) { if (!o[k]) o[k] = {}; o = o[k]; }
  o[ks[ks.length - 1]] = value;
}
function debounce(fn, ms) { let t = null; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
const isDark = () => window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// ------------------------------------------------------------------ tabs
function initTabs() {
  $$('.tab').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
}
function showTab(name) {
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tabpanel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  if (name === 'results') loadResults();
  if (name === 'analysis') runAnalysis();
  if (name === 'controllers') renderControllerCards();
  window.dispatchEvent(new Event('resize'));
}

// ------------------------------------------------------------------ init
async function init() {
  initTabs();
  try {
    const [info, ctrls, presets] = await Promise.all([api('/api/info'), api('/api/controllers'), api('/api/scenarios')]);
    S.info = info;
    S.controllers = ctrls.controllers;
    S.ctrlErrors = ctrls.errors || [];
    S.customDir = ctrls.custom_dir;
    S.presets = presets.presets || [];
  } catch (e) {
    toast('サーバーに接続できません: ' + e.message, true);
    return;
  }
  buildBackendSelect();
  buildPerceptionSelect();
  buildPresetSelect();
  buildControllerList();
  const first = S.presets.find(p => p.name === S.info.default_scenario.name) || S.info.default_scenario;
  loadScenario(first);
  bindScenarioInputs();
  $('#presetLoad').addEventListener('click', () => {
    const p = S.presets.find(x => x.name === $('#presetSel').value);
    if (p) { loadScenario(p); toast(`プリセット「${p.name}」を読み込みました`); }
  });
  $('#presetSave').addEventListener('click', savePreset);
  $('#presetDelete').addEventListener('click', deletePreset);
  $('#startBtn').addEventListener('click', startRun);
  $('#stopBtn').addEventListener('click', stopRun);
  $('#resReload').addEventListener('click', loadResults);
  $('#resFilter').addEventListener('input', renderResults);
  $('#resToAnalysis').addEventListener('click', () => {
    if (!S.selection.size) { toast('試行を選択してください', true); return; }
    showTab('analysis');
  });
  $('#groupBy').addEventListener('change', runAnalysis);
  $('#anaExport').addEventListener('click', exportZip);
  $('#anaPng').addEventListener('click', openComparePng);
  buildMetricSelect();
  window.addEventListener('resize', debounce(() => drawLive(), 100));
  $('#showScan').addEventListener('change', () => drawLive());
  $('#showCands').addEventListener('change', () => drawLive());
  startStream();
  drawLive();
}

// ------------------------------------------------------------------ selects
function buildBackendSelect() {
  const sel = $('#backendSel');
  sel.innerHTML = '';
  for (const b of S.info.backends) sel.append(el('option', { value: b.name }, b.label));
  sel.value = S.info.default_backend;
  sel.addEventListener('change', updateBackendHint);
  updateBackendHint();
}
function updateBackendHint() {
  const name = $('#backendSel').value;
  const b = (S.live && S.live.backends ? S.live.backends : S.info.backends).find(x => x.name === name);
  $('#simSpeedWrap').style.display = name === 'sim2d' ? '' : 'none';
  $('#backendHint').textContent = b ? (b.ok ? '利用可能: ' : '利用不可: ') + (b.message || '') : '';
}
function buildPerceptionSelect() {
  const sel = $('#perception');
  for (const [k, v] of Object.entries(S.info.perception_modes)) sel.append(el('option', { value: k }, v));
  sel.value = 'scan_tracking';
}
function buildPresetSelect() {
  const sel = $('#presetSel');
  const cur = sel.value;
  sel.innerHTML = '';
  for (const p of S.presets) sel.append(el('option', { value: p.name }, `${p.name}${p.description ? ' — ' + p.description : ''}`));
  if (cur) sel.value = cur;
}
function buildMetricSelect() {
  const sel = $('#anaMetrics');
  const defaults = ['travel_time', 'min_clearance', 'personal_space_time', 'min_ttc', 'collisions', 'compute_ms_mean'];
  sel.append(el('option', { value: 'success_rate', selected: true }, '成功率 [%]'));
  for (const [k, [label, unit]] of Object.entries(S.info.metric_info)) {
    sel.append(el('option', { value: k, selected: defaults.includes(k) }, `${label} [${unit}]`));
  }
  sel.addEventListener('change', () => renderAnalysisBars());
}

// ------------------------------------------------------------------ scenario form
function loadScenario(sc) {
  S.scenario = JSON.parse(JSON.stringify(sc));
  $('#scName').value = S.scenario.name || '';
  $('#scDesc').value = S.scenario.description || '';
  for (const inp of $$('[data-k]', $('#tab-run'))) {
    if (inp.closest('.human-card')) continue;
    const v = getPath(S.scenario, inp.dataset.k);
    if (inp.type === 'checkbox') inp.checked = !!v; else inp.value = v;
  }
  if (S.presets.find(p => p.name === S.scenario.name)) $('#presetSel').value = S.scenario.name;
  buildHumanCards();
  refreshPreview();
}
function bindScenarioInputs() {
  for (const inp of $$('[data-k]', $('#tab-run'))) {
    if (inp.closest('.human-card')) continue;
    inp.addEventListener('change', () => {
      setPath(S.scenario, inp.dataset.k, inp.type === 'checkbox' ? inp.checked : parseFloat(inp.value));
      refreshPreview();
    });
  }
  $('#scName').addEventListener('change', () => { S.scenario.name = $('#scName').value; });
  $('#scDesc').addEventListener('change', () => { S.scenario.description = $('#scDesc').value; });
}
const refreshPreview = debounce(async () => {
  try {
    S.preview = await api('/api/scenario/preview', { body: { scenario: S.scenario } });
  } catch (e) { S.preview = null; }
  drawLive();
}, 150);

const HUMAN_FIELDS = {
  common: [['speed', '歩行速度 [m/s]', 0.1], ['radius', '半径 (評価用) [m]', 0.05]],
  crossing: [['cross_x', '横切る位置 x [m]', 0.5], ['lateral_start', '開始時の横距離 [m]', 0.5],
    ['lateral_end', '終了時の横距離 [m]', 0.5], ['angle_deg', '交差角 [deg] (90=直角)', 5]],
  head_on: [['start_x', '開始位置 x [m]', 0.5], ['lateral_offset', '経路からの横ずれ y [m]', 0.1],
    ['travel_distance', '歩く距離 [m]', 0.5]],
  same_direction: [['start_x', '開始位置 x [m]', 0.5], ['lateral_offset', '経路からの横ずれ y [m]', 0.1],
    ['travel_distance', '歩く距離 [m]', 0.5]],
  static: [['x', 'x [m]', 0.1], ['y', 'y [m]', 0.1]],
};
function buildHumanCards() {
  const wrap = $('#humanCards');
  wrap.innerHTML = '';
  S.scenario.humans.forEach((h, i) => wrap.append(humanCard(h, i)));
}
function humanCard(h, i) {
  const color = HUMAN_COLORS[i % HUMAN_COLORS.length];
  const card = el('div', { class: 'card human-card' + (h.enabled ? '' : ' disabled') });
  const upd = (k, v) => { S.scenario.humans[i][k] = v; refreshPreview(); };
  const numInput = (k, label, step) => el('label', {}, label,
    el('input', { type: 'number', step, value: h[k], onchange: e => upd(k, parseFloat(e.target.value)) }));
  const enabled = el('input', { type: 'checkbox', checked: h.enabled, onchange: e => {
    upd('enabled', e.target.checked); card.classList.toggle('disabled', !e.target.checked); } });
  const head = el('div', { class: 'head' },
    el('h2', {}, el('span', { class: 'swatch', style: `background:${color}` }), `人 ${i + 1}`),
    el('label', { class: 'check small' }, enabled, '有効'));
  const patSel = el('select', {});
  for (const [k, v] of Object.entries(S.info.patterns)) patSel.append(el('option', { value: k }, v));
  patSel.value = h.pattern;
  const groups = {};
  const body = el('div', { class: 'body' });
  body.append(el('div', { class: 'row' }, el('label', { class: 'grow' }, '移動方法', patSel)));
  body.append(el('div', { class: 'row' }, ...HUMAN_FIELDS.common.map(f => numInput(...f))));
  // crossing
  const side = el('select', { onchange: e => upd('from_side', e.target.value) },
    el('option', { value: 'left' }, '左から (+y 側)'), el('option', { value: 'right' }, '右から (−y 側)'));
  side.value = h.from_side;
  groups.crossing = el('div', { class: 'pattern-fields' },
    el('div', { class: 'row' }, el('label', {}, '出発する側', side), numInput(...HUMAN_FIELDS.crossing[0])),
    el('div', { class: 'row' }, ...HUMAN_FIELDS.crossing.slice(1).map(f => numInput(...f))));
  groups.head_on = el('div', { class: 'pattern-fields' }, el('div', { class: 'row' }, ...HUMAN_FIELDS.head_on.map(f => numInput(...f))));
  groups.same_direction = el('div', { class: 'pattern-fields' }, el('div', { class: 'row' }, ...HUMAN_FIELDS.same_direction.map(f => numInput(...f))));
  groups.static = el('div', { class: 'pattern-fields' }, el('div', { class: 'row' }, ...HUMAN_FIELDS.static.map(f => numInput(...f))));
  const wpText = el('textarea', { rows: 3, onchange: e => {
    const pts = e.target.value.split(/[\n;]+/).map(s => s.split(/[ ,]+/).filter(Boolean).map(Number))
      .filter(p => p.length >= 2 && p.every(Number.isFinite)).map(p => [p[0], p[1]]);
    upd('waypoints', pts);
  } });
  wpText.value = (h.waypoints || []).map(p => `${p[0]}, ${p[1]}`).join('\n');
  groups.waypoints = el('div', { class: 'pattern-fields' },
    el('label', {}, 'ウェイポイント (1行に "x, y"、先頭が初期位置)', wpText),
    el('label', { class: 'check small' }, el('input', { type: 'checkbox', checked: h.loop, onchange: e => upd('loop', e.target.checked) }), '繰り返す'));
  for (const g of Object.values(groups)) body.append(g);
  // trigger
  const trig = el('select', {});
  for (const [k, v] of Object.entries(S.info.triggers)) trig.append(el('option', { value: k }, v));
  trig.value = h.trigger;
  const tTime = numInput('trigger_time', '開始時刻 [s]', 0.5);
  const tX = numInput('trigger_robot_x', 'ロボット通過 x [m]', 0.5);
  const tOff = numInput('encounter_offset', '到達時刻のずらし [s] (+で人が遅れる)', 0.25);
  body.append(el('h3', {}, '歩き始めるタイミング'));
  body.append(el('div', { class: 'row' }, el('label', { class: 'grow' }, 'トリガー', trig)));
  const trigRow = el('div', { class: 'row' }, tTime, tX, tOff);
  body.append(trigRow);
  body.append(el('label', { class: 'check' }, el('input', { type: 'checkbox', checked: h.reactive, onchange: e => upd('reactive', e.target.checked) }),
    'ロボットを避ける (Social Force 風の反応)'));
  const sync = () => {
    for (const [k, g] of Object.entries(groups)) g.hidden = k !== patSel.value;
    tTime.style.display = trig.value === 'time' || (trig.value === 'encounter' && patSel.value !== 'crossing') ? '' : 'none';
    tX.style.display = trig.value === 'robot_x' ? '' : 'none';
    tOff.style.display = trig.value === 'encounter' && patSel.value === 'crossing' ? '' : 'none';
  };
  patSel.addEventListener('change', () => { upd('pattern', patSel.value); sync(); });
  trig.addEventListener('change', () => { upd('trigger', trig.value); sync(); });
  sync();
  card.append(head, body);
  return card;
}
async function savePreset() {
  S.scenario.name = $('#scName').value || 'scenario';
  S.scenario.description = $('#scDesc').value || '';
  try {
    const r = await api('/api/scenarios', { body: { scenario: S.scenario } });
    S.presets = (await api('/api/scenarios')).presets;
    buildPresetSelect();
    $('#presetSel').value = r.name;
    toast(`プリセット「${r.name}」を保存しました`);
  } catch (e) { toast('保存に失敗: ' + e.message, true); }
}
async function deletePreset() {
  const name = $('#presetSel').value;
  if (!name || !confirm(`プリセット「${name}」を削除しますか?`)) return;
  try {
    await api('/api/scenarios/' + encodeURIComponent(name), { method: 'DELETE' });
    S.presets = (await api('/api/scenarios')).presets;
    buildPresetSelect();
    toast('削除しました');
  } catch (e) { toast('削除に失敗: ' + e.message, true); }
}

// ------------------------------------------------------------------ controllers
function buildControllerList() {
  const wrap = $('#ctrlList');
  wrap.innerHTML = '';
  for (const c of S.controllers) {
    if (!S.ctrlState[c.name]) S.ctrlState[c.name] = { checked: c.builtin && ['dwa', 'mppi', 'mpc'].includes(c.name), params: {} };
    const st = S.ctrlState[c.name];
    const cb = el('input', { type: 'checkbox', checked: st.checked, onchange: e => { st.checked = e.target.checked; } });
    const tag = c.kind === 'external' ? '外部ノード' : (c.builtin ? '組込み' : '自作');
    const item = el('div', { class: 'ctrl-item' },
      el('div', { class: 'top' }, cb,
        el('span', { class: 'name', style: `color:${ctrlColor(c.name, S.controllers.map(x => x.name))}` }, c.label),
        el('span', { class: 'tag' }, tag), el('span', { class: 'tag' }, c.name)));
    if (c.params && c.params.length) {
      const grid = el('div', { class: 'param-grid' });
      for (const p of c.params) grid.append(paramInput(p, st.params));
      item.append(el('details', {}, el('summary', {}, `パラメータ (${c.params.length})`),
        el('p', { class: 'hint' }, c.description || ''), grid,
        el('button', { class: 'btn ghost small', onclick: () => { st.params = {}; buildControllerList(); } }, '既定値に戻す')));
    }
    wrap.append(item);
  }
}
function paramInput(p, store) {
  const val = store[p.key] !== undefined ? store[p.key] : p.default;
  const title = p.help ? `${p.key}: ${p.help}` : p.key;
  if (p.type === 'bool') {
    return el('label', { class: 'check', title }, el('input', { type: 'checkbox', checked: !!val,
      onchange: e => { store[p.key] = e.target.checked; } }), p.label);
  }
  if (p.type === 'choice') {
    const s = el('select', { onchange: e => { store[p.key] = e.target.value; } }, ...(p.choices || []).map(c => el('option', { value: c }, c)));
    s.value = val;
    return el('label', { title }, p.label, s);
  }
  return el('label', { title }, p.label, el('input', { type: p.type === 'str' ? 'text' : 'number',
    step: p.step ?? 'any', min: p.min ?? null, max: p.max ?? null, value: val,
    onchange: e => { store[p.key] = p.type === 'str' ? e.target.value : parseFloat(e.target.value); } }));
}
function renderControllerCards() {
  $('#customDirHint').innerHTML = `自作の制御器は <code>${S.customDir || 'custom_controllers/'}</code> に .py (Python) または .yaml (外部ROS 2ノード) を置くと自動で追加されます。下の「再読込」で反映。`;
  const errs = $('#ctrlErrors');
  errs.innerHTML = '';
  for (const e of S.ctrlErrors) errs.append(el('div', { class: 'err' }, `${e.file}\n${e.error}`));
  const wrap = $('#ctrlCards');
  wrap.innerHTML = '';
  if (!$('#ctrlReload')) {
    $('#ctrlErrors').before(el('div', { class: 'row' }, el('button', { id: 'ctrlReload', class: 'btn ghost', onclick: async () => {
      const r = await api('/api/controllers');
      S.controllers = r.controllers; S.ctrlErrors = r.errors || [];
      buildControllerList(); renderControllerCards(); toast('制御器を再読込しました');
    } }, '再読込')));
  }
  for (const c of S.controllers) {
    const rows = (c.params || []).map(p => el('tr', {},
      el('td', {}, p.label, el('div', { class: 'pkey' }, p.key)), el('td', { class: 'num' }, String(p.default))));
    wrap.append(el('div', { class: 'ctrl-card' },
      el('h4', { style: `color:${ctrlColor(c.name, S.controllers.map(x => x.name))}` }, c.label),
      el('div', { class: 'hint' }, `${c.name} · ${c.kind === 'external' ? '外部ノード' : (c.builtin ? '組込み' : '自作')} · ${c.source}`),
      el('p', {}, c.description || ''), el('table', {}, ...rows)));
  }
}

// ------------------------------------------------------------------ run / stop
async function startRun() {
  const ctrls = S.controllers.filter(c => S.ctrlState[c.name] && S.ctrlState[c.name].checked)
    .map(c => ({ name: c.name, params: S.ctrlState[c.name].params }));
  if (!ctrls.length) { toast('制御器を選択してください', true); return; }
  S.scenario.name = $('#scName').value || S.scenario.name;
  S.scenario.description = $('#scDesc').value || '';
  const body = {
    backend: $('#backendSel').value,
    name: $('#batchName').value || S.scenario.name,
    scenarios: [S.scenario],
    controllers: ctrls,
    repeats: parseInt($('#repeats').value || '1', 10),
    perception: $('#perception').value,
    use_prediction: $('#usePrediction').checked,
    scan_inflation: parseFloat($('#scanInflation').value || '0.15'),
    sim_speed: parseFloat($('#simSpeed').value || '0'),
  };
  $('#startBtn').disabled = true;
  try {
    const r = await api('/api/run', { body });
    toast(`評価を開始しました: ${r.batch.id} (${r.batch.trials.length} 試行)`);
  } catch (e) {
    toast('開始できません: ' + e.message, true);
  } finally {
    setTimeout(() => { $('#startBtn').disabled = false; }, 800);
  }
}
async function stopRun() {
  try { await api('/api/stop', { body: {} }); toast('停止を要求しました'); } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ live stream
function startStream() {
  const es = new EventSource('/api/stream');
  es.onmessage = ev => {
    try { S.live = JSON.parse(ev.data); } catch (e) { return; }
    onLive();
  };
  es.onerror = () => {
    renderPills(null);
  };
}
let lastRecentKey = '';
function onLive() {
  const L = S.live;
  renderPills(L.backends);
  updateBackendHint();
  const st = L.status || {};
  const tr = st.trial;
  // trails
  const tid = tr ? tr.id : null;
  if (tid !== S.trails.id) { S.trails = { id: tid, robot: [], humans: [] }; }
  if (tr && tr.state === 'running' && L.snap && L.snap.robot) {
    const r = L.snap.robot;
    pushTrail(S.trails.robot, [r.x, r.y]);
    (L.snap.humans || []).forEach((h, i) => {
      if (!S.trails.humans[i]) S.trails.humans[i] = [];
      pushTrail(S.trails.humans[i], [h.x, h.y]);
    });
  }
  renderStats();
  drawLive();
  const key = JSON.stringify((st.last_runs || []).map(r => r.run_id));
  if (key !== lastRecentKey) {
    lastRecentKey = key;
    renderRecent(st.last_runs || []);
    if ($('#tab-results').classList.contains('active')) loadResults();
  }
}
function pushTrail(arr, p) {
  const last = arr[arr.length - 1];
  if (!last || Math.hypot(last[0] - p[0], last[1] - p[1]) > 0.03) arr.push(p);
  if (arr.length > 3000) arr.shift();
}
function renderPills(backends) {
  const wrap = $('#backendPills');
  wrap.innerHTML = '';
  if (!backends) { wrap.append(el('span', { class: 'pill bad' }, el('span', { class: 'dot' }), 'サーバー切断')); return; }
  for (const b of backends) {
    const cls = b.busy ? 'busy' : (b.ok ? 'ok' : 'bad');
    wrap.append(el('span', { class: `pill ${cls}`, title: b.message || '' }, el('span', { class: 'dot' }),
      `${b.name === 'gazebo' ? 'Gazebo' : '2D'}: ${b.busy ? '実行中' : (b.ok ? '待機' : '未接続')}`));
  }
}
function stat(k, v, cls = '') { return el('div', { class: 'stat' }, el('div', { class: 'k' }, k), el('div', { class: 'v ' + cls }, v)); }
function renderStats() {
  const L = S.live;
  const wrap = $('#liveStats');
  wrap.innerHTML = '';
  const st = L.status || {};
  const tr = st.trial;
  const b = st.batch;
  const snap = L.snap || {};
  const r = snap.robot || {};
  let nearest = null;
  if (r && snap.humans && L.scene && L.scene.humans_cfg) {
    snap.humans.forEach((h, i) => {
      const cfg = L.scene.humans_cfg[i];
      if (!cfg || !cfg.enabled) return;
      const d = Math.hypot(h.x - r.x, h.y - r.y) - (cfg.radius || 0.25) - (r.radius || 0.25);
      if (nearest === null || d < nearest) nearest = d;
    });
  }
  const dbg = (L.scene && L.scene.debug_info) || {};
  wrap.append(
    stat('状態', tr ? (STATE_LABEL[tr.state] || tr.state) : (b && b.status === 'running' ? '準備中' : '待機')),
    stat('試行', tr ? `${tr.controller}` : '-'),
    stat('経過時間 [s]', tr ? fmt(tr.elapsed, 1) : '-'),
    stat('速度 v [m/s]', fmt(r.v, 2)),
    stat('角速度 ω [rad/s]', fmt(r.w, 2)),
    stat('人とのクリアランス [m]', nearest === null ? '-' : fmt(nearest, 2), nearest !== null && nearest < 0.3 ? 'bad' : ''),
    stat('最小クリアランス [m]', tr && tr.min_clearance !== null ? fmt(tr.min_clearance, 2) : '-',
      tr && tr.min_clearance !== null && tr.min_clearance <= 0 ? 'bad' : ''),
    stat('接触', tr ? String(tr.collisions) : '-', tr && tr.collisions ? 'bad' : ''),
    stat('計算時間 [ms]', fmt(dbg.compute_ms, 1)),
    stat('バッチ進捗', b ? `${b.done} / ${b.total}` : '-'),
  );
  const pct = b && b.total ? 100 * (b.done + (tr && tr.state === 'running' ? 0.5 : 0)) / b.total : 0;
  $('#progressBar').style.width = `${Math.min(100, pct)}%`;
  $('#liveMsg').textContent = b ? `バッチ ${b.id} — ${b.status}` + (tr ? ` / ${tr.id}` : '') : '「評価開始」で選択した制御器 × 繰り返し回数の試行を順番に実行します。';
}
function badge(outcome) { return el('span', { class: `badge ${outcome}` }, OUTCOME_LABEL[outcome] || outcome); }
function renderRecent(runs) {
  const t = $('#recentTbl');
  t.innerHTML = '';
  t.append(el('tr', {}, el('th', { class: 'l' }, '制御器'), el('th', { class: 'l' }, '結果'), el('th', {}, '移動時間[s]'),
    el('th', {}, '最小クリアランス[m]'), el('th', {}, 'PS侵入[s]')));
  for (const s of runs.slice().reverse()) {
    t.append(el('tr', {}, el('td', { class: 'l' }, s.controller), el('td', { class: 'l' }, badge(s.outcome)),
      el('td', { class: 'num' }, fmt(s.travel_time)), el('td', { class: 'num' }, fmt(s.min_clearance)),
      el('td', { class: 'num' }, fmt(s.personal_space_time))));
  }
}

// ------------------------------------------------------------------ live canvas
function drawLive() {
  const cv = $('#liveCanvas');
  if (!cv || !S.info) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 900;
  const world = S.info.world;
  const L = S.live || {};
  const scene = (L.scene && L.scene.course) ? L.scene : null;
  const course = scene ? scene.course : (S.scenario ? S.scenario.course : { start_x: 0, goal_x: 10, y: 0, robot_start_offset: 1, goal_extension: 2 });
  const halfW = world.walls ? world.corridor_width / 2 : 4;
  let x0 = course.start_x - (course.robot_start_offset || 0) - 2.0;
  let x1 = course.goal_x + Math.max(course.goal_extension || 0, 1) + 1.0;
  if (!(L.status && L.status.trial) && S.preview && S.preview.routes) {
    for (const rt of S.preview.routes) for (const p of (rt || [])) { x0 = Math.min(x0, p[0] - 1); x1 = Math.max(x1, p[0] + 1); }
  }
  x0 = Math.max(x0, world.x_min - 1); x1 = Math.min(x1, world.x_max + 1);
  const y0 = -halfW - 0.4, y1 = halfW + 0.4;
  const aspect = (y1 - y0) / (x1 - x0);
  const cssH = Math.max(220, Math.min(520, cssW * aspect));
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  cv.style.height = cssH + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sc = Math.min(cssW / (x1 - x0), cssH / (y1 - y0));
  const ox = (cssW - sc * (x1 - x0)) / 2, oy = (cssH - sc * (y1 - y0)) / 2;
  const P = (x, y) => [ox + (x - x0) * sc, cssH - oy - (y - y0) * sc];
  ctx.fillStyle = cssVar('--canvas-bg'); ctx.fillRect(0, 0, cssW, cssH);
  // grid
  ctx.strokeStyle = cssVar('--floor-line'); ctx.lineWidth = 1;
  for (let gx = Math.ceil(x0); gx <= x1; gx++) { const [a, b] = P(gx, y0); const [, d] = P(gx, y1); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(a, d); ctx.stroke(); }
  for (let gy = Math.ceil(y0); gy <= y1; gy++) { const [a, b] = P(x0, gy); const [c] = P(x1, gy); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, b); ctx.stroke(); }
  // walls
  if (world.walls) {
    ctx.strokeStyle = cssVar('--wall'); ctx.lineWidth = 4;
    for (const yy of [halfW, -halfW]) { const [a, b] = P(x0, yy); const [c] = P(x1, yy); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, b); ctx.stroke(); }
  }
  // START / GOAL
  const line = (x, color, label) => {
    const [a, b] = P(x, -halfW); const [, d] = P(x, halfW);
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(a, d); ctx.stroke();
    ctx.fillStyle = color; ctx.font = '600 12px system-ui'; ctx.textAlign = 'center'; ctx.fillText(label, a, d - 4);
  };
  line(course.start_x, '#1a9e5c', 'START');
  line(course.goal_x, '#d1453b', 'GOAL');
  // path
  const path = scene ? scene.path : (S.preview ? S.preview.path : null);
  if (path) {
    ctx.setLineDash([6, 5]); ctx.strokeStyle = isDark() ? '#5b6678' : '#9aa5b8'; ctx.lineWidth = 1.5;
    ctx.beginPath(); path.forEach((p, i) => { const [a, b] = P(p[0], p[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
    ctx.setLineDash([]);
  }
  const running = L.status && L.status.trial;
  // planned human routes (scenario preview, when idle)
  if (!running && S.preview && S.preview.routes) {
    S.preview.routes.forEach((rt, i) => {
      if (!rt) return;
      ctx.strokeStyle = HUMAN_COLORS[i % HUMAN_COLORS.length]; ctx.globalAlpha = 0.7; ctx.setLineDash([4, 4]); ctx.lineWidth = 2;
      ctx.beginPath(); rt.forEach((p, k) => { const [a, b] = P(p[0], p[1]); k ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
      ctx.setLineDash([]);
      const [sx, sy] = P(rt[0][0], rt[0][1]);
      const rad = (S.scenario.humans[i].radius || 0.25) * sc;
      ctx.beginPath(); ctx.arc(sx, sy, rad, 0, 2 * Math.PI); ctx.fillStyle = HUMAN_COLORS[i % HUMAN_COLORS.length]; ctx.globalAlpha = 0.25; ctx.fill();
      ctx.globalAlpha = 1;
      const e = rt[rt.length - 1]; const [ex, ey] = P(e[0], e[1]);
      drawArrowHead(ctx, rt.length > 1 ? P(rt[rt.length - 2][0], rt[rt.length - 2][1]) : [sx, sy], [ex, ey], HUMAN_COLORS[i % HUMAN_COLORS.length]);
    });
    if (S.preview.robot_start) {
      const [rx, ry] = P(S.preview.robot_start[0], S.preview.robot_start[1]);
      ctx.beginPath(); ctx.arc(rx, ry, S.info.robot.radius * sc, 0, 2 * Math.PI); ctx.strokeStyle = cssVar('--robot'); ctx.lineWidth = 2; ctx.stroke();
    }
  }
  if (!L.snap) return;
  const snap = L.snap;
  if (!running) {
    ctx.fillStyle = cssVar('--muted'); ctx.font = '12px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('待機中 — 破線はシナリオの予定経路、○ は開始位置', 8, cssH - 8);
    return;
  }
  ctx.globalAlpha = 1;
  const baseAlpha = ctx.globalAlpha;
  // scan points
  if ($('#showScan').checked && snap.scan && snap.scan.length) {
    ctx.fillStyle = isDark() ? 'rgba(220,225,235,.55)' : 'rgba(60,70,90,.45)';
    for (const p of snap.scan) { const [a, b] = P(p[0], p[1]); ctx.fillRect(a - 1, b - 1, 2, 2); }
  }
  // trails
  const drawTrail = (pts, color) => {
    if (pts.length < 2) return;
    ctx.strokeStyle = color; ctx.globalAlpha = 0.55 * baseAlpha; ctx.lineWidth = 2;
    ctx.beginPath(); pts.forEach((p, i) => { const [a, b] = P(p[0], p[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
    ctx.globalAlpha = baseAlpha;
  };
  drawTrail(S.trails.robot, cssVar('--robot'));
  S.trails.humans.forEach((t, i) => drawTrail(t || [], HUMAN_COLORS[i % HUMAN_COLORS.length]));
  // controller debug trajectories
  if (scene) {
    if ($('#showCands').checked && scene.candidates) {
      ctx.strokeStyle = 'rgba(18,163,127,.25)'; ctx.lineWidth = 1;
      for (const c of scene.candidates) {
        if (!c || c.length < 2) continue;
        ctx.beginPath(); c.forEach((p, i) => { const [a, b] = P(p[0], p[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
      }
    }
    if (scene.trajectory && scene.trajectory.length > 1) {
      ctx.strokeStyle = cssVar('--traj'); ctx.lineWidth = 2.5;
      ctx.beginPath(); scene.trajectory.forEach((p, i) => { const [a, b] = P(p[0], p[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke();
    }
  }
  // humans
  const hcfg = scene ? scene.humans_cfg : null;
  (snap.humans || []).forEach((h, i) => {
    const cfg = hcfg ? hcfg[i] : (S.scenario ? S.scenario.humans[i] : null);
    if (cfg && !cfg.enabled) return;
    const [a, b] = P(h.x, h.y);
    const rad = ((cfg && cfg.radius) || 0.25) * sc;
    ctx.beginPath(); ctx.arc(a, b, rad, 0, 2 * Math.PI);
    ctx.fillStyle = HUMAN_COLORS[i % HUMAN_COLORS.length]; ctx.globalAlpha = 0.85 * baseAlpha; ctx.fill(); ctx.globalAlpha = baseAlpha;
    ctx.fillStyle = '#fff'; ctx.font = '600 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i + 1), a, b);
    ctx.textBaseline = 'alphabetic';
    const sp = Math.hypot(h.vx, h.vy);
    if (sp > 0.05) {
      const [c, d] = P(h.x + h.vx * 0.8, h.y + h.vy * 0.8);
      ctx.strokeStyle = HUMAN_COLORS[i % HUMAN_COLORS.length]; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
      drawArrowHead(ctx, [a, b], [c, d], HUMAN_COLORS[i % HUMAN_COLORS.length]);
    }
  });
  // robot
  const r = snap.robot;
  if (r) {
    const [a, b] = P(r.x, r.y);
    const rad = (r.radius || S.info.robot.radius) * sc;
    ctx.beginPath(); ctx.arc(a, b, rad, 0, 2 * Math.PI);
    ctx.fillStyle = cssVar('--robot'); ctx.globalAlpha = 0.85 * baseAlpha; ctx.fill(); ctx.globalAlpha = baseAlpha;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(a + Math.cos(r.yaw) * rad, b - Math.sin(r.yaw) * rad); ctx.stroke();
    // nearest human distance
    if (hcfg) {
      let best = null;
      (snap.humans || []).forEach((h, i) => {
        if (!hcfg[i] || !hcfg[i].enabled) return;
        const d = Math.hypot(h.x - r.x, h.y - r.y);
        if (!best || d < best.d) best = { d, h, i };
      });
      if (best && best.d < 2.5) {
        const [c, d] = P(best.h.x, best.h.y);
        ctx.strokeStyle = best.d < 1.2 ? '#d1453b' : '#c98a00'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = ctx.strokeStyle; ctx.font = '600 11px system-ui'; ctx.textAlign = 'left';
        ctx.fillText(`${best.d.toFixed(2)} m`, (a + c) / 2 + 4, (b + d) / 2 - 4);
      }
    }
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = cssVar('--muted'); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left';
  ctx.fillText(`t_sim = ${fmt(snap.time, 2)} s`, 8, cssH - 8);
}
function drawArrowHead(ctx, from, to, color) {
  const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const s = 7;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - s * Math.cos(ang - 0.4), to[1] - s * Math.sin(ang - 0.4));
  ctx.lineTo(to[0] - s * Math.cos(ang + 0.4), to[1] - s * Math.sin(ang + 0.4));
  ctx.closePath(); ctx.fill();
}

// ------------------------------------------------------------------ results
async function loadResults() {
  try {
    const r = await api('/api/results');
    S.batches = r.batches || [];
  } catch (e) { toast('結果を取得できません: ' + e.message, true); return; }
  renderResults();
}
function renderResults() {
  const wrap = $('#batchList');
  wrap.innerHTML = '';
  const q = ($('#resFilter').value || '').toLowerCase();
  if (!S.batches.length) { wrap.append(el('p', { class: 'hint' }, 'まだ結果がありません。')); return; }
  for (const b of S.batches) {
    const runs = (b.runs || []).filter(r => !q || [b.name, b.id, r.controller, r.scenario, r.perception].join(' ').toLowerCase().includes(q));
    if (q && !runs.length) continue;
    const allRefs = runs.map(r => `${b.id}/${r.run_id}`);
    const selAll = el('input', { type: 'checkbox', checked: allRefs.length && allRefs.every(x => S.selection.has(x)),
      onchange: e => { allRefs.forEach(x => e.target.checked ? S.selection.add(x) : S.selection.delete(x)); renderResults(); } });
    const del = el('button', { class: 'btn ghost small', onclick: async ev => {
      ev.stopPropagation();
      if (!confirm(`バッチ ${b.id} を削除しますか? (元に戻せません)`)) return;
      try { await api('/api/results/' + encodeURIComponent(b.id), { method: 'DELETE' }); allRefs.forEach(x => S.selection.delete(x)); loadResults(); }
      catch (e) { toast('削除できません: ' + e.message, true); }
    } }, '削除');
    const head = el('div', { class: 'bhead' },
      el('label', { class: 'check small' }, selAll, el('span', { class: 'bname' }, b.name)),
      el('span', { class: 'bmeta' }, `${b.id} · ${b.created} · ${b.backend || ''} · ${b.status} · ${runs.length}/${(b.trials || []).length} 試行`), del);
    const t = el('table', { class: 'tbl' });
    t.append(el('tr', {}, el('th', {}, ''), el('th', { class: 'l' }, '試行'), el('th', { class: 'l' }, '制御器'), el('th', { class: 'l' }, 'シナリオ'),
      el('th', { class: 'l' }, '認識'), el('th', { class: 'l' }, '結果'), el('th', {}, '移動時間[s]'), el('th', {}, '最小距離[m]'),
      el('th', {}, '最小クリアランス[m]'), el('th', {}, 'PS侵入[s]'), el('th', {}, '最小TTC[s]'), el('th', {}, '接触')));
    for (const r of runs) {
      const ref = `${b.id}/${r.run_id}`;
      const cb = el('input', { type: 'checkbox', checked: S.selection.has(ref), onclick: e => e.stopPropagation(),
        onchange: e => { e.target.checked ? S.selection.add(ref) : S.selection.delete(ref); } });
      t.append(el('tr', { class: 'clickable' + (S.detailRef === ref ? ' selected' : ''), onclick: () => showRunDetail(b.id, r.run_id) },
        el('td', {}, cb), el('td', { class: 'l num' }, r.run_id), el('td', { class: 'l' }, r.controller), el('td', { class: 'l' }, r.scenario),
        el('td', { class: 'l' }, r.perception || ''), el('td', { class: 'l' }, badge(r.outcome)),
        el('td', { class: 'num' }, fmt(r.travel_time)), el('td', { class: 'num' }, fmt(r.min_center_distance)),
        el('td', { class: 'num' }, fmt(r.min_clearance)), el('td', { class: 'num' }, fmt(r.personal_space_time)),
        el('td', { class: 'num' }, fmt(r.min_ttc)), el('td', { class: 'num' }, String(r.collisions ?? '-'))));
    }
    wrap.append(el('div', { class: 'batch' }, head, el('div', { class: 'table-wrap' }, t)));
  }
}
function xyRanges(traces, pad = 0.6) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const t of traces) {
    (t.x || []).forEach(v => { if (v !== null) { x0 = Math.min(x0, v); x1 = Math.max(x1, v); } });
    (t.y || []).forEach(v => { if (v !== null) { y0 = Math.min(y0, v); y1 = Math.max(y1, v); } });
  }
  if (!Number.isFinite(x0)) return {};
  const ylo = Math.min(y0, -1.5) - pad, yhi = Math.max(y1, 1.5) + pad;
  const w = Math.max(320, (document.querySelector('main') || document.body).clientWidth - 80);
  const height = Math.max(280, Math.min(600, w * (yhi - ylo) / (x1 - x0 + 2 * pad) + 120));
  return {
    height,
    xaxis: { title: { text: 'x [m]' }, range: [x0 - pad, x1 + pad], constrain: 'domain' },
    yaxis: { title: { text: 'y [m]' }, range: [ylo, yhi], scaleanchor: 'x', scaleratio: 1, constrain: 'domain' },
  };
}
function plotLayout(title, xt, yt, extra = {}) {
  const dark = isDark();
  return Object.assign({
    title: { text: title, font: { size: 14 } }, margin: { l: 55, r: 15, t: 40, b: 45 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: dark ? '#e5eaf2' : '#1b2230', size: 11 },
    xaxis: { title: { text: xt }, gridcolor: dark ? '#2a3446' : '#e3e8f0', zeroline: false },
    yaxis: { title: { text: yt }, gridcolor: dark ? '#2a3446' : '#e3e8f0', zeroline: false },
    legend: { orientation: 'h', y: -0.22 },
  }, extra);
}
const PLOT_CFG = { responsive: true, displaylogo: false, toImageButtonOptions: { format: 'png', scale: 2 } };
async function showRunDetail(batchId, runId) {
  S.detailRef = `${batchId}/${runId}`;
  renderResults();
  let d;
  try { d = await api(`/api/results/${encodeURIComponent(batchId)}/${encodeURIComponent(runId)}`); }
  catch (e) { toast(e.message, true); return; }
  const box = $('#runDetail');
  box.hidden = false;
  const s = d.summary;
  $('#runTitle').textContent = `${runId} — ${s.controller_label || s.controller} / ${s.scenario}`;
  const dl = $('#runDownloads');
  dl.innerHTML = '';
  for (const [k, lab] of [['trajectory', '軌跡 PNG'], ['distance', '距離 PNG'], ['clearance', 'クリアランス PNG'], ['velocity', '速度 PNG']]) {
    dl.append(el('a', { class: 'btn ghost small', href: `/api/figure.png?kind=${k}&runs=${encodeURIComponent(S.detailRef)}`, target: '_blank' }, lab));
  }
  const mg = $('#runMetrics');
  mg.innerHTML = '';
  mg.append(stat('結果', OUTCOME_LABEL[s.outcome] || s.outcome, s.outcome === 'goal' ? 'good' : 'bad'));
  for (const [k, [label, unit]] of Object.entries(S.info.metric_info)) {
    if (s[k] === undefined || s[k] === null) continue;
    mg.append(stat(`${label} [${unit}]`, typeof s[k] === 'number' ? fmt(s[k], unit === '回' ? 0 : 2) : String(s[k])));
  }
  const ts = d.timeseries;
  const sc = d.config.scenario;
  const humans = sc.humans.map((h, i) => ({ h, i })).filter(o => o.h.enabled && ts[`h${o.i + 1}_x`]);
  // trajectory
  const traces = [{ x: ts.x, y: ts.y, mode: 'lines', name: 'ロボット', line: { color: '#1f6feb', width: 3 } }];
  for (const { i } of humans) traces.push({ x: ts[`h${i + 1}_x`], y: ts[`h${i + 1}_y`], mode: 'lines', name: `人${i + 1}`, line: { color: HUMAN_COLORS[i], dash: 'dash' } });
  const W = S.info.world.corridor_width / 2;
  const shapes = [
    { type: 'line', x0: sc.course.start_x, x1: sc.course.start_x, y0: -W, y1: W, line: { color: '#1a9e5c', width: 3 } },
    { type: 'line', x0: sc.course.goal_x, x1: sc.course.goal_x, y0: -W, y1: W, line: { color: '#d1453b', width: 3 } },
  ];
  for (const hd of (s.humans || [])) {
    if (!hd.robot_at_min) continue;
    shapes.push({ type: 'line', x0: hd.robot_at_min[0], y0: hd.robot_at_min[1], x1: hd.human_at_min[0], y1: hd.human_at_min[1], line: { color: '#888', width: 1, dash: 'dot' } });
  }
  Plotly.react('plotTraj', traces, plotLayout('軌跡 (x-y)', 'x [m]', 'y [m]', Object.assign({ shapes }, xyRanges(traces))), PLOT_CFG);
  // distance
  const rr = d.config.robot_radius || 0.25;
  const dt = [];
  for (const { h, i } of humans) {
    const xs = ts[`h${i + 1}_x`], ys = ts[`h${i + 1}_y`];
    dt.push({ x: ts.t, y: ts.t.map((_, k) => (xs[k] === null ? null : Math.hypot(xs[k] - ts.x[k], ys[k] - ts.y[k]))), mode: 'lines', name: `人${i + 1} 中心間距離`, line: { color: HUMAN_COLORS[i] } });
    dt.push({ x: ts.t, y: ts.t.map((_, k) => (xs[k] === null ? null : Math.hypot(xs[k] - ts.x[k], ys[k] - ts.y[k]) - rr - h.radius)), mode: 'lines', name: `人${i + 1} クリアランス`, line: { color: HUMAN_COLORS[i], dash: 'dot' } });
  }
  const ev = sc.evaluation;
  const tmax = ts.t.length ? ts.t[ts.t.length - 1] : 1;
  Plotly.react('plotDist', dt, plotLayout('人との距離', '時間 [s]', '距離 [m]', {
    shapes: [
      { type: 'line', x0: 0, x1: tmax, y0: ev.personal_space, y1: ev.personal_space, line: { color: '#c98a00', dash: 'dash', width: 1 } },
      { type: 'line', x0: 0, x1: tmax, y0: ev.intimate_space, y1: ev.intimate_space, line: { color: '#d1453b', dash: 'dash', width: 1 } },
      { type: 'line', x0: 0, x1: tmax, y0: 0, y1: 0, line: { color: '#d1453b', width: 1 } },
    ],
  }), PLOT_CFG);
  Plotly.react('plotVel', [
    { x: ts.t, y: ts.v, name: 'v 実測', line: { color: '#1f6feb' } },
    { x: ts.t, y: ts.cmd_v, name: 'v 指令', line: { color: '#1f6feb', dash: 'dot' } },
    { x: ts.t, y: ts.w, name: 'ω 実測', yaxis: 'y2', line: { color: '#e377c2' } },
    { x: ts.t, y: ts.cmd_w, name: 'ω 指令', yaxis: 'y2', line: { color: '#e377c2', dash: 'dot' } },
  ], plotLayout('速度', '時間 [s]', 'v [m/s]', { margin: { l: 55, r: 60, t: 40, b: 45 }, yaxis2: { title: { text: 'ω [rad/s]' }, overlaying: 'y', side: 'right', showgrid: false } }), PLOT_CFG);
  box.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ------------------------------------------------------------------ analysis
function selectedRefs() { return Array.from(S.selection).map(x => { const [b, r] = x.split('/'); return { batch_id: b, run_id: r }; }); }
async function runAnalysis() {
  const refs = selectedRefs();
  $('#anaSel').textContent = refs.length ? `${refs.length} 試行を選択中` : '「結果」タブで試行を選択してください。';
  if (!refs.length) { S.analysis = null; renderAnalysis(); return; }
  try {
    S.analysis = await api('/api/analysis', { body: { runs: refs, group_by: $('#groupBy').value } });
  } catch (e) { toast(e.message, true); return; }
  renderAnalysis();
  renderOverlays(refs);
}
function renderAnalysis() {
  const t = $('#anaTable');
  t.innerHTML = '';
  if (!S.analysis) { $('#anaBars').innerHTML = ''; return; }
  const keys = ['travel_time', 'min_clearance', 'min_center_distance', 'personal_space_time', 'intimate_space_time', 'min_ttc', 'max_lateral_deviation', 'collisions', 'rms_jerk', 'compute_ms_mean'];
  const mi = S.info.metric_info;
  t.append(el('tr', {}, el('th', { class: 'l' }, 'グループ'), el('th', {}, '試行'), el('th', {}, '成功率'), ...keys.map(k => el('th', {}, `${mi[k][0]} [${mi[k][1]}]`))));
  for (const g of S.analysis.groups) {
    t.append(el('tr', {}, el('td', { class: 'l', style: `color:${ctrlColor(g.controller || '', S.controllers.map(c => c.name))}` }, g.group),
      el('td', { class: 'num' }, String(g.n)), el('td', { class: 'num' }, `${Math.round(100 * g.success_rate)}%`),
      ...keys.map(k => el('td', { class: 'num' }, g[k] ? `${fmt(g[k].mean)} ± ${fmt(g[k].std)}` : '-'))));
  }
  renderAnalysisBars();
}
function renderAnalysisBars() {
  const wrap = $('#anaBars');
  wrap.innerHTML = '';
  if (!S.analysis) return;
  const metrics = Array.from($('#anaMetrics').selectedOptions).map(o => o.value);
  const groups = S.analysis.groups;
  const names = S.controllers.map(c => c.name);
  for (const m of metrics) {
    const div = el('div', { class: 'plot' });
    wrap.append(div);
    const y = groups.map(g => (m === 'success_rate' ? 100 * g.success_rate : (g[m] ? g[m].mean : null)));
    const err = groups.map(g => (m === 'success_rate' ? 0 : (g[m] ? g[m].std : 0)));
    const label = m === 'success_rate' ? ['成功率', '%'] : S.info.metric_info[m];
    Plotly.newPlot(div, [{
      type: 'bar', x: groups.map(g => g.group), y, error_y: { type: 'data', array: err, visible: true },
      marker: { color: groups.map(g => ctrlColor(g.controller || '', names)) },
      text: y.map(v => (v === null ? '' : v.toFixed(2))), textposition: 'outside', hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>',
    }], plotLayout(`${label[0]} [${label[1]}]`, '', label[1], { showlegend: false }), PLOT_CFG);
  }
}
async function renderOverlays(refs) {
  const take = refs.slice(0, 12);
  const runs = [];
  for (const r of take) {
    try { runs.push(await api(`/api/results/${encodeURIComponent(r.batch_id)}/${encodeURIComponent(r.run_id)}`)); } catch (e) { /* skip */ }
  }
  const names = S.controllers.map(c => c.name);
  const tTraj = [], tDist = [], tSpeed = [];
  runs.forEach((d, k) => {
    const s = d.summary, ts = d.timeseries, sc = d.config.scenario;
    const color = ctrlColor(s.controller, names);
    const lab = `${s.controller} / ${s.scenario} #${s.repeat || 1}`;
    tTraj.push({ x: ts.x, y: ts.y, mode: 'lines', name: lab, line: { color, width: 2 } });
    sc.humans.forEach((h, i) => {
      if (!h.enabled || !ts[`h${i + 1}_x`]) return;
      if (k === 0) tTraj.push({ x: ts[`h${i + 1}_x`], y: ts[`h${i + 1}_y`], mode: 'lines', name: `人${i + 1} (${lab})`, line: { color: HUMAN_COLORS[i], dash: 'dash', width: 1 } });
    });
    const rr = d.config.robot_radius || 0.25;
    const clr = ts.t.map((_, j) => {
      let best = null;
      sc.humans.forEach((h, i) => {
        const xs = ts[`h${i + 1}_x`];
        if (!h.enabled || !xs || xs[j] === null) return;
        const v = Math.hypot(xs[j] - ts.x[j], ts[`h${i + 1}_y`][j] - ts.y[j]) - rr - h.radius;
        if (best === null || v < best) best = v;
      });
      return best;
    });
    tDist.push({ x: ts.t, y: clr, mode: 'lines', name: lab, line: { color } });
    tSpeed.push({ x: ts.t, y: ts.v, mode: 'lines', name: lab, line: { color } });
  });
  Plotly.react('anaTraj', tTraj, plotLayout(`軌跡の重ね描き (最大12試行)`, 'x [m]', 'y [m]', xyRanges(tTraj)), PLOT_CFG);
  Plotly.react('anaDist', tDist, plotLayout('最も近い人とのクリアランス', '時間 [s]', 'クリアランス [m]'), PLOT_CFG);
  Plotly.react('anaSpeed', tSpeed, plotLayout('速度', '時間 [s]', 'v [m/s]'), PLOT_CFG);
}
async function exportZip() {
  const refs = selectedRefs();
  if (!refs.length) { toast('試行を選択してください', true); return; }
  toast('レポートを生成しています…');
  try {
    const r = await api('/api/export', { body: { runs: refs, group_by: $('#groupBy').value }, raw: true });
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const m = /filename="([^"]+)"/.exec(cd);
    const a = el('a', { href: URL.createObjectURL(blob), download: m ? m[1] : 'eval_report.zip' });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  } catch (e) { toast('出力に失敗: ' + e.message, true); }
}
function openComparePng() {
  const refs = Array.from(S.selection);
  if (!refs.length) { toast('試行を選択してください', true); return; }
  const metrics = Array.from($('#anaMetrics').selectedOptions).map(o => o.value).join(',');
  window.open(`/api/compare.png?runs=${encodeURIComponent(refs.join(','))}&metrics=${encodeURIComponent(metrics)}&group_by=${$('#groupBy').value}`, '_blank');
}

window.addEventListener('DOMContentLoaded', init);
