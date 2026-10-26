/** Bounded, fair polling for single-camera and preview-grid views. */
export class FrameScheduler {
  constructor({fetchFrame, onFrame, onError = () => {}, concurrency = 4, timeout = 5000}) {
    this.fetchFrame = fetchFrame;
    this.onFrame = onFrame;
    this.onError = onError;
    this.concurrency = concurrency;
    this.timeout = timeout;
    this.tasks = new Map();
    this.active = 0;
    this.cursor = 0;
    this.paused = false;
    this.disposed = false;
    this.timer = null;
  }

  setTargets(targets) {
    const wanted = new Map(targets.map(target => [target.key, target]));
    for (const [key, task] of this.tasks) {
      if (!wanted.has(key) || wanted.get(key).quality !== task.quality) {
        task.controller?.abort();
        this.tasks.delete(key);
      }
    }
    for (const [key, target] of wanted) {
      const existing = this.tasks.get(key);
      if (existing) {
        if (existing.interval !== target.interval) existing.nextDue = 0;
        existing.interval = target.interval;
      } else {
        this.tasks.set(key, {...target, nextDue: 0, controller: null});
      }
    }
    this.kick();
  }

  setPaused(paused) {
    if (this.paused === paused) return;
    this.paused = paused;
    if (paused) {
      clearTimeout(this.timer);
      this.timer = null;
      for (const task of this.tasks.values()) task.controller?.abort();
    } else {
      for (const task of this.tasks.values()) task.nextDue = 0;
      this.kick();
    }
  }

  kick(delay = 0) {
    if (this.paused || this.disposed) return;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.tick(), delay);
  }

  tick() {
    this.timer = null;
    if (this.paused || this.disposed || this.active >= this.concurrency) return;
    const tasks = [...this.tasks.values()];
    const now = performance.now();
    const start = this.cursor % Math.max(1, tasks.length);
    for (let n = 0; n < tasks.length && this.active < this.concurrency; n++) {
      const index = (start + n) % tasks.length;
      const task = tasks[index];
      if (task.controller || task.nextDue > now) continue;
      this.cursor = (index + 1) % tasks.length;
      this.launch(task);
    }
    if (this.active < this.concurrency) {
      const waiting = tasks.filter(task => !task.controller);
      if (waiting.length) {
        const due = Math.min(...waiting.map(task => task.nextDue));
        this.kick(Math.max(10, due - performance.now()));
      }
    }
  }

  async launch(task) {
    const controller = new AbortController();
    task.controller = controller;
    task.nextDue = performance.now() + task.interval;
    this.active++;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeout);
    const current = () => !this.disposed && !this.paused && this.tasks.get(task.key) === task;
    try {
      const data = await this.fetchFrame(task, controller.signal);
      if (current() && !controller.signal.aborted) this.onFrame(task, data);
    } catch (error) {
      if (current() && (!controller.signal.aborted || timedOut)) {
        task.nextDue = performance.now() + Math.max(1000, task.interval);
        this.onError(task, timedOut ? new Error('응답 시간 초과 · 재연결 중') : error);
      }
    } finally {
      clearTimeout(timer);
      task.controller = null;
      this.active--;
      this.kick();
    }
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.timer);
    for (const task of this.tasks.values()) task.controller?.abort();
    this.tasks.clear();
  }
}

const unavailable = camera => ['MISSING', 'OFFLINE', 'ERROR'].includes(camera.state);
export function frameTargets({status, mode, selection, quality, gridPc, gridFps}) {
  if (!status) return [];
  const limit = Math.max(0.1, Number(status.max_fps) || 5);
  const fps = mode === 'grid' ? Math.min(limit, 5, Math.max(1, Number(gridFps) || 2)) : limit;
  const targets = [];
  for (const pc of status.pcs) {
    if (!pc.online || (mode === 'grid' && gridPc !== 'all' && pc.pc !== gridPc)) continue;
    for (const camera of pc.cameras) {
      const key = `${pc.pc}/${camera.name}`;
      if (unavailable(camera) || (mode !== 'grid' && key !== selection)) continue;
      targets.push({key, pc: pc.pc, serial: String(camera.name),
        quality: mode === 'grid' ? 'preview' : quality, interval: 1000 / fps});
    }
  }
  return targets;
}

export function startViewer(doc = document) {
  const get = id => doc.getElementById(id);
  const view = get('view'), camera = get('camera'), quality = get('quality');
  const gridPc = get('grid-pc'), columns = get('columns'), gridFps = get('grid-fps');
  const grid = get('grid'), image = get('image'), notice = get('notice');
  const params = new URLSearchParams(location.search);
  view.value = params.get('view') === 'grid' ? 'grid' : 'single';
  gridPc.value = 'all';
  if (['auto', '2', '3', '4', '6'].includes(params.get('cols'))) columns.value = params.get('cols');
  if (['1', '2', '5'].includes(params.get('grid_fps'))) gridFps.value = params.get('grid_fps');
  const tiles = new Map(), models = new Map();
  let status = null, selection = '', paused = false, statusError = '', optionSignature = '';
  let pcSignature = '', initialSelection = true, statusTimer = null;

  const model = key => {
    if (!models.has(key)) models.set(key, {count: null, progress: 0, lastImage: 0, url: null, quality: null, error: ''});
    return models.get(key);
  };
  const find = key => {
    const [pcName, serial] = key.split('/');
    const pc = status?.pcs.find(item => item.pc === pcName);
    return {pc, cam: pc?.cameras.find(item => String(item.name) === serial)};
  };

  function overlay(key) {
    const {pc, cam} = find(key), state = model(key);
    if (paused) return '일시정지';
    if (statusError) return statusError;
    if (!pc?.online) return 'Capture PC 연결 끊김 · 재연결 중';
    if (!cam || cam.state === 'MISSING') return '연결되지 않은 카메라';
    if (unavailable(cam)) return `카메라 상태: ${cam.state}`;
    if (state.progress && performance.now() - state.progress > 3000) return '프레임 갱신 대기 · 촬영 전환 / 트리거 대기';
    if (state.error) return '영상 대기 중 · ' + state.error;
    const expectedQuality = view.value === 'grid' ? 'preview' : quality.value;
    return state.url && state.quality === expectedQuality ? '' : '영상 연결 중…';
  }

  function setNotice(element, text) { element.textContent = text; element.hidden = !text; }

  function renderCamera(key) {
    const {pc, cam} = find(key), state = model(key), tile = tiles.get(key);
    if (tile) {
      const text = overlay(key);
      tile.node.classList.toggle('offline', !pc?.online || !cam || unavailable(cam));
      tile.node.classList.toggle('live', pc?.online && cam && !unavailable(cam) && !text);
      tile.foot.textContent = `${pc?.state || 'offline'} / ${cam?.state || 'MISSING'}`;
      tile.count.textContent = `frames ${state.count ?? '—'}`;
      tile.image.hidden = !state.url || state.quality !== 'preview' || cam?.state === 'MISSING';
      if (!tile.image.hidden && tile.image.getAttribute('src') !== state.url) tile.image.src = state.url;
      setNotice(tile.notice, text);
    }
    if (key === selection && view.value === 'single') {
      image.hidden = !state.url || state.quality !== quality.value || cam?.state === 'MISSING';
      if (!image.hidden && image.getAttribute('src') !== state.url) image.src = state.url;
      setNotice(notice, overlay(key));
    }
  }

  function updateURL() {
    const url = new URL(location.href);
    url.searchParams.set('view', view.value);
    if (selection) {
      const [pc, serial] = selection.split('/');
      url.searchParams.set('pc', pc); url.searchParams.set('camera', serial);
    }
    url.searchParams.set('grid_pc', gridPc.value);
    url.searchParams.set('cols', columns.value);
    url.searchParams.set('grid_fps', gridFps.value);
    history.replaceState(null, '', url);
  }

  const scheduler = new FrameScheduler({
    async fetchFrame(task, signal) {
      const response = await fetch(`/api/frame/${encodeURIComponent(task.pc)}/${encodeURIComponent(task.serial)}?quality=${task.quality}`, {signal});
      if (!response.ok) throw new Error((await response.json()).error || '영상을 기다리는 중');
      return response.blob();
    },
    onFrame(task, blob) {
      const state = model(task.key), old = state.url;
      state.url = URL.createObjectURL(blob);
      state.quality = task.quality; state.lastImage = performance.now(); state.error = '';
      renderCamera(task.key);
      if (old) URL.revokeObjectURL(old);
    },
    onError(task, error) { model(task.key).error = error.message; renderCamera(task.key); },
  });

  function targets() {
    return frameTargets({status, mode: view.value, selection, quality: quality.value,
      gridPc: gridPc.value, gridFps: Number(gridFps.value)});
  }

  function refresh() {
    const isGrid = view.value === 'grid';
    get('single').hidden = isGrid; grid.hidden = !isGrid;
    for (const id of ['camera-control', 'quality-control']) get(id).hidden = isGrid;
    for (const id of ['pc-control', 'columns-control', 'rate-control']) get(id).hidden = !isGrid;
    grid.classList.toggle('manual-columns', columns.value !== 'auto');
    grid.style.setProperty('--columns', columns.value === 'auto' ? '4' : columns.value);
    let visible = 0;
    for (const [key, tile] of tiles) {
      tile.node.hidden = gridPc.value !== 'all' && !key.startsWith(gridPc.value + '/');
      if (!tile.node.hidden) visible++;
      renderCamera(key);
    }
    get('empty').hidden = !isGrid || visible > 0;
    get('empty').textContent = statusError || (!status ? '카메라 목록을 불러오는 중…' : '선택한 그룹에 카메라가 없습니다.');
    const requests = targets();
    scheduler.setTargets(requests);
    scheduler.setPaused(paused || doc.hidden);
    if (isGrid) {
      const fps = Math.min(Number(gridFps.value), status?.max_fps || 5, 5);
      get('details').textContent = `그리드 · ${requests.length}대 연결 / ${visible}대 · 미리보기 · 최대 ${fps} fps`;
      get('hint').textContent = '카메라를 클릭하면 단일 화면으로 확대합니다. 촬영 전환 시 영상이 잠시 멈출 수 있습니다.';
    } else {
      const {pc, cam} = find(selection), state = model(selection);
      get('details').textContent = cam ? `${selection.replace('/', ' / ')} · ${pc.backend || '—'} · ${pc.state} / ${cam.state} · frames ${state.count ?? '—'}` : '카메라를 선택하세요';
      get('hint').textContent = '촬영 준비·전환 중에는 영상이 잠시 멈출 수 있습니다.';
      if (selection) renderCamera(selection);
    }
  }

  function createTile(pc, cam) {
    const key = `${pc.pc}/${cam.name}`, node = doc.createElement('button');
    node.type = 'button'; node.className = 'tile';
    node.setAttribute('aria-label', `${pc.pc} 카메라 ${cam.name} 확대`);
    const head = doc.createElement('div'); head.className = 'tile-head';
    const serial = doc.createElement('strong'); serial.textContent = cam.name;
    const name = doc.createElement('span'); name.textContent = pc.pc;
    head.append(serial, name);
    const picture = doc.createElement('div'); picture.className = 'tile-picture';
    const img = doc.createElement('img'); img.alt = `${pc.pc} / ${cam.name}`; img.hidden = true;
    const message = doc.createElement('span'); message.className = 'notice';
    picture.append(img, message);
    const foot = doc.createElement('div'); foot.className = 'tile-foot';
    const dot = doc.createElement('span'); dot.className = 'dot';
    const state = doc.createElement('span'), count = doc.createElement('span');
    foot.append(dot, state, count); node.append(head, picture, foot);
    node.onclick = () => { camera.value = key; selection = key; view.value = 'single'; updateURL(); refresh(); };
    tiles.set(key, {node, image: img, notice: message, foot: state, count});
    return node;
  }

  function updateOptions() {
    const signature = JSON.stringify(status.pcs.map(pc => [pc.pc, pc.cameras.map(cam => [cam.name, unavailable(cam) ? cam.state : ''])]));
    if (signature !== optionSignature) {
      optionSignature = signature;
      const previous = camera.value; camera.replaceChildren();
      const ordered = [], keys = new Set();
      for (const pc of status.pcs) {
        const group = doc.createElement('optgroup'); group.label = pc.pc;
        for (const cam of pc.cameras) {
          const key = `${pc.pc}/${cam.name}`, option = doc.createElement('option');
          option.value = key; option.textContent = cam.name + (unavailable(cam) ? ' · ' + cam.state : '');
          group.append(option); keys.add(key);
          ordered.push(tiles.get(key)?.node || createTile(pc, cam));
        }
        camera.append(group);
      }
      for (const [key] of tiles) if (!keys.has(key)) {
        const old = models.get(key); if (old?.url) URL.revokeObjectURL(old.url);
        tiles.delete(key); models.delete(key);
      }
      grid.replaceChildren(...ordered);
      const values = [...camera.options].map(option => option.value);
      const wanted = values.find(key => key.endsWith('/' + params.get('camera')) && (!params.get('pc') || key.startsWith(params.get('pc') + '/')));
      const firstOnline = status.pcs.flatMap(pc => pc.cameras.map(cam => ({pc, cam}))).find(({pc, cam}) => pc.online && !unavailable(cam));
      camera.value = initialSelection && wanted ? wanted : values.includes(previous) ? previous : firstOnline ? `${firstOnline.pc.pc}/${firstOnline.cam.name}` : values[0] || '';
      selection = camera.value; initialSelection = false;
    }
    const names = status.pcs.map(pc => pc.pc), signaturePc = JSON.stringify(names);
    if (signaturePc !== pcSignature) {
      const previous = gridPc.value; const all = doc.createElement('option');
      all.value = 'all'; all.textContent = '전체 Capture PC'; gridPc.replaceChildren(all);
      for (const name of names) { const option = doc.createElement('option'); option.value = name; option.textContent = name; gridPc.append(option); }
      const desired = pcSignature ? previous : params.get('grid_pc');
      gridPc.value = names.includes(desired) ? desired : 'all'; pcSignature = signaturePc;
    }
    quality.options[0].textContent = `미리보기 · 640px / 최대 ${status.max_fps} fps`;
    quality.options[1].textContent = `원본 · 2048px / 최대 ${status.max_fps} fps`;
    get('pcs').replaceChildren(...status.pcs.map(pc => {
      const button = doc.createElement('button'); button.type = 'button'; button.className = 'pc' + (pc.online ? ' on' : '');
      button.textContent = `${pc.pc} · ${pc.online ? pc.cameras.filter(cam => !unavailable(cam)).length + ' cams · ' + pc.state : 'offline'}`;
      button.onclick = () => { view.value = 'grid'; gridPc.value = pc.pc; updateURL(); refresh(); };
      return button;
    }));
  }

  async function pollStatus() {
    try {
      const response = await fetch('/api/cameras', {signal: AbortSignal.timeout(10000)});
      if (!response.ok) throw new Error('상태 조회 실패');
      status = await response.json(); statusError = ''; updateOptions();
      const now = performance.now();
      for (const pc of status.pcs) for (const cam of pc.cameras) {
        const state = model(`${pc.pc}/${cam.name}`), count = cam.frame_count ?? cam.frame_id;
        if (count !== undefined && count !== state.count) { state.count = count; state.progress = now; }
      }
    } catch (error) {
      status = null; statusError = '뷰어 서버 연결 끊김 · 재연결 중';
      if (!selection) setNotice(notice, statusError);
    }
    refresh();
    statusTimer = setTimeout(pollStatus, 1000);
  }

  view.onchange = () => { updateURL(); refresh(); };
  camera.onchange = () => { selection = camera.value; updateURL(); refresh(); };
  quality.onchange = refresh;
  for (const control of [gridPc, columns, gridFps]) control.onchange = () => { updateURL(); refresh(); };
  get('pause').onclick = () => {
    paused = !paused; get('pause').textContent = paused ? '재생' : '일시정지'; refresh();
  };
  doc.addEventListener('visibilitychange', () => scheduler.setPaused(paused || doc.hidden));
  window.addEventListener('beforeunload', () => {
    clearTimeout(statusTimer); scheduler.dispose();
    for (const state of models.values()) if (state.url) URL.revokeObjectURL(state.url);
  });
  refresh(); pollStatus();
  return scheduler;
}

if (typeof document !== 'undefined') startViewer();
