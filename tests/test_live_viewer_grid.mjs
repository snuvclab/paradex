import test from 'node:test';
import assert from 'node:assert/strict';
import {FrameScheduler, frameTargets} from '../paradex/io/camera_system/templates/live_viewer.mjs';

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, message = 'condition') {
  const deadline = performance.now() + 1000;
  while (!predicate()) {
    if (performance.now() > deadline) assert.fail('Timed out: ' + message);
    await sleep(5);
  }
}
const target = (key, quality = 'preview') => ({key, pc: 'capture12', serial: key, quality, interval: 1000});

test('grid selects online preview cameras only, with PC filtering and rate limits', () => {
  const status = {max_fps: 5, pcs: [
    {pc: 'capture12', online: true, cameras: [{name: '1', state: 'CAPTURING'}, {name: 'missing', state: 'MISSING'}]},
    {pc: 'capture13', online: true, cameras: [{name: '2', state: 'CAPTURING'}, {name: 'bad', state: 'ERROR'}]},
    {pc: 'capture14', online: false, cameras: [{name: '3', state: 'CAPTURING'}]},
  ]};
  const options = {status, mode: 'grid', quality: 'full', gridPc: 'all', gridFps: 2};
  assert.deepEqual(frameTargets(options).map(t => [t.key, t.quality, t.interval]), [
    ['capture12/1', 'preview', 500], ['capture13/2', 'preview', 500],
  ]);
  assert.deepEqual(frameTargets({...options, gridPc: 'capture13'}).map(t => t.key), ['capture13/2']);
  assert.equal(frameTargets({...options, gridFps: 30})[0].interval, 200);
  assert.deepEqual(frameTargets({...options, mode: 'single', selection: 'capture13/2'}).map(t => [t.key, t.quality]), [['capture13/2', 'full']]);
  assert.deepEqual(frameTargets({...options, status: null}), []);
});

test('23 cameras are polled fairly with at most four requests in flight', async t => {
  const pending = [], started = [], frames = [];
  let concurrent = 0, peak = 0;
  const scheduler = new FrameScheduler({
    fetchFrame(task, signal) {
      concurrent++; peak = Math.max(peak, concurrent); started.push(task.key);
      return new Promise((resolve, reject) => {
        pending.push(() => resolve(task.key));
        signal.addEventListener('abort', () => reject(new Error('aborted')), {once: true});
      }).finally(() => concurrent--);
    },
    onFrame(task, value) { frames.push(value); },
  });
  t.after(() => scheduler.dispose());
  scheduler.setTargets(Array.from({length: 23}, (_, i) => target(String(i))));
  await until(() => started.length === 4);
  assert.equal(concurrent, 4);
  while (frames.length < 23) {
    const batch = pending.splice(0); batch.forEach(resolve => resolve());
    await until(() => frames.length === 23 || pending.length > 0);
  }
  assert.equal(peak, 4);
  assert.equal(new Set(started).size, 23);
  assert.equal(started.length, 23);
});

test('switching grid to a single camera cancels removed requests and ignores late frames', async t => {
  const started = [], frames = [];
  const scheduler = new FrameScheduler({
    fetchFrame(task, signal) {
      return new Promise(resolve => started.push({task, signal, resolve}));
    },
    onFrame(task, data) { frames.push([task.key, task.quality, data]); },
  });
  t.after(() => scheduler.dispose());
  scheduler.setTargets([target('one'), target('two')]);
  await until(() => started.length === 2);
  scheduler.setTargets([target('one', 'full')]);
  assert.ok(started.every(request => request.signal.aborted));
  started.slice(0, 2).forEach(request => request.resolve('old preview'));
  await until(() => started.length === 3);
  started[2].resolve('new full frame');
  await until(() => frames.length === 1);
  assert.deepEqual(frames, [['one', 'full', 'new full frame']]);
});

test('pause aborts requests and resume restarts without publishing paused frames', async t => {
  const started = [], frames = [];
  const scheduler = new FrameScheduler({
    fetchFrame(task, signal) { return new Promise(resolve => started.push({signal, resolve})); },
    onFrame(task, value) { frames.push(value); },
  });
  t.after(() => scheduler.dispose());
  scheduler.setTargets([target('one')]);
  await until(() => started.length === 1);
  scheduler.setPaused(true);
  assert.ok(started[0].signal.aborted);
  started[0].resolve('paused');
  await sleep(30);
  assert.equal(started.length, 1); assert.deepEqual(frames, []);
  scheduler.setPaused(false);
  await until(() => started.length === 2);
  started[1].resolve('resumed');
  await until(() => frames.length === 1);
  assert.deepEqual(frames, ['resumed']);
});

test('one slow camera times out independently without starving healthy cameras', async t => {
  const frames = [], errors = [], attempts = [];
  const scheduler = new FrameScheduler({timeout: 30, concurrency: 2,
    fetchFrame(task, signal) {
      attempts.push(task.key);
      if (task.key !== 'slow') return Promise.resolve('jpeg');
      return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')), {once: true}));
    },
    onFrame(task) { frames.push(task.key); }, onError(task) { errors.push(task.key); },
  });
  t.after(() => scheduler.dispose());
  scheduler.setTargets([target('slow'), target('good1'), target('good2'), target('good3')]);
  await until(() => frames.length === 3 && errors.length === 1);
  assert.deepEqual(errors, ['slow']);
  assert.deepEqual(new Set(frames), new Set(['good1', 'good2', 'good3']));
  assert.equal(attempts.filter(key => key === 'slow').length, 1);
});
