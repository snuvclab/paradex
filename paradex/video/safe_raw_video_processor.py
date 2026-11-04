"""Verified NAS uploads; delete matching local videos only after verified publication.

Encode on captures1, validate every output frame, copy to a hidden NAS staging
file, verify SHA-256, preserve any old destination, then atomically publish.
Damaged/truncated inputs go to recovery/videos rather than the normal videos.
Use --keep-raw to retain local videos even after successful verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time
import uuid

GIB = 1024 ** 3


def signature(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def confined_file(path, root):
    relative = path.relative_to(root)
    if path.resolve(strict=True) != root.resolve() / relative or not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('Refusing symlink/non-regular video: ' + str(path))


def has_open_writer(path):
    """Match device/inode too, so a writer through another name is protected."""
    expected = signature(path)[:2]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.geteuid():
                continue
            fds = list((process / 'fd').iterdir())
        except PermissionError:
            # Other users' processes are not the capture user's recording daemons.
            continue
        except FileNotFoundError:
            continue
        for fd in fds:
            try:
                st = fd.stat()
                if [st.st_dev, st.st_ino] != expected:
                    continue
                values = dict(line.split(':', 1) for line in (process / 'fdinfo' / fd.name).read_text().splitlines())
                if (int(values['flags'].strip(), 8) & os.O_ACCMODE) in (os.O_WRONLY, os.O_RDWR):
                    return True
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return False


def probe(path, decode=False):
    count = 'nb_read_frames' if decode else 'nb_read_packets'
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-count_frames' if decode else '-count_packets',
               '-show_entries', 'stream=codec_name,width,height,nb_frames,{},avg_frame_rate'.format(count),
               '-of', 'json', str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result.returncode or (decode and result.stderr.strip()):
        raise RuntimeError('Video validation failed: ' + result.stderr[-1500:])
    streams = json.loads(result.stdout).get('streams', [])
    if len(streams) != 1:
        raise RuntimeError('Expected one video stream: ' + str(path))
    stream = streams[0]
    n = stream.get(count)
    if n in (None, 'N/A') or int(n) <= 0:
        raise RuntimeError('No readable frames/packets: ' + str(path))
    stream['count'] = int(n)
    declared = stream.get('nb_frames')
    stream['declared_frames'] = int(declared) if declared not in (None, 'N/A') else None
    stream['diagnostics'] = result.stderr[-1500:]
    return stream


def reference_timestamps(episode):
    """Metadata only: timestamp receive gaps must not be mistaken for video loss."""
    try:
        import numpy as np
        path = episode / 'raw/timestamps/frame_id.npy'
        if not path.exists():
            return None
        ids = np.load(path, allow_pickle=False).astype('int64')
        if ids.ndim != 1 or not len(ids):
            return {'error': 'Empty or invalid timestamp frame IDs'}
        steps = np.diff(ids)
        return {'count': len(ids), 'first_id': int(ids[0]), 'last_id': int(ids[-1]),
                'span': int(ids[-1] - ids[0] + 1),
                'missing_ids': int(np.maximum(steps - 1, 0).sum()),
                'non_increasing': int((steps <= 0).sum())}
    except Exception as exc:
        return {'error': str(exc)}


class SafeUploader:
    def __init__(self, home=None, nas=None, roots=None, scratch=None, state=None,
                 workers=2, publish=True, reserve_bytes=20 * GIB, keep_raw=False):
        self.home = Path(home or Path.home())
        self.nas = Path(nas or self.home / 'shared_data')
        self.roots = [Path(p) for p in (roots or [self.home / 'captures1', self.home / 'captures2'])]
        self.scratch = Path(scratch or self.home / 'captures1/.paradex_safe_upload')
        self.state = Path(state or self.home / '.local/state/paradex/safe-upload')
        self.run = datetime.now().strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        self.run_dir = self.state / 'runs' / self.run
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.workers, self.publish = workers, publish
        self.reserve_bytes = reserve_bytes
        self.jobs, self.lock = {}, threading.RLock()
        self.stop = threading.Event()
        self.space_lock = threading.Condition()
        self.reserved = 0
        self.publisher = None
        self.keep_raw = bool(keep_raw)
        self.journal_lock = threading.Lock()

    def update(self, key, **values):
        with self.lock:
            self.jobs[key].update(values, updated_at=time.time())

    def snapshot(self):
        with self.lock:
            jobs = {key: dict(value) for key, value in self.jobs.items()}
        summary = {name: sum(job['status'] == name for job in jobs.values())
                   for name in ('pending', 'processing', 'uploading', 'completed', 'failed', 'paused')}
        return {'run': self.run, 'host': self.home.name, 'raw_deletion_enabled': not self.keep_raw,
                'deleted_files': sum(j.get('deleted_files', 0) for j in jobs.values()),
                'deleted_bytes': sum(j.get('deleted_bytes', 0) for j in jobs.values()),
                'workers': self.workers, 'summary': summary, 'jobs': jobs}

    def save(self):
        data = self.snapshot()
        write_json(self.run_dir / 'progress.json', data)
        write_json(self.state / 'latest.json', {'run': self.run, 'run_dir': str(self.run_dir),
                                              'summary': data['summary'], 'raw_deletion_enabled': data['raw_deletion_enabled'],
                                              'deleted_files': data['deleted_files'], 'deleted_bytes': data['deleted_bytes'],
                                              'updated_at': time.time()})
        if self.publisher:
            metadata = [{'name': key, **value} for key, value in data['jobs'].items()]
            try:
                self.publisher.send_multipart([
                    b'data', json.dumps({'items': metadata, 'timestamp': datetime.now().isoformat(),
                                        'publisher': 'safe_video_processor'}).encode()
                ], flags=1)
            except Exception:
                pass

    @contextmanager
    def space(self, amount):
        self.scratch.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 180
        with self.space_lock:
            while True:
                if self.stop.is_set():
                    raise InterruptedError('Uploader paused')
                stat = os.statvfs(str(self.scratch))
                available = stat.f_bavail * stat.f_frsize
                if available - self.reserved >= amount + self.reserve_bytes:
                    self.reserved += amount
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Not enough captures1 scratch space; original retained')
                self.space_lock.wait(2)
        try:
            yield
        finally:
            with self.space_lock:
                self.reserved -= amount
                self.space_lock.notify_all()

    def check_stop(self):
        if self.stop.is_set():
            raise InterruptedError('Uploader paused; original retained')

    def encode(self, source, output, log, strict):
        self.check_stop()
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning', '-y']
        if strict:
            command += ['-xerror', '-err_detect', 'explode']
        command += ['-threads', '2', '-i', str(source), '-map', '0:v:0', '-an',
                    '-c:v', 'mjpeg', '-q:v', '3', '-threads', '2', '-vsync', '0', str(output)]
        with log.open('wb') as handle:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=handle,
                                       start_new_session=True)
            started = time.monotonic()
            while process.poll() is None:
                if self.stop.wait(0.5) or time.monotonic() - started > 3600:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise InterruptedError('Encoding stopped; original retained')
        return process.returncode

    def publish_file(self, source, stage, target, expected_hash):
        """Existing NAS data is preserved even when replacing a bad prior upload."""
        self.check_stop()
        result = subprocess.run(['rsync', '-t', '--partial', '--inplace', '--checksum',
                                 '--no-owner', '--no-group', '--', str(source), str(stage)],
                                capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError('NAS transfer failed; originals retained: ' + result.stderr[-1000:])
        verified_signature = signature(stage)
        if file_hash(stage) != expected_hash or signature(stage) != verified_signature:
            raise RuntimeError('NAS checksum mismatch; original and old destination retained')
        self.check_stop()
        old = None
        if target.exists():
            old = target.parent / '.safe-upload-backups' / self.run / target.name
            old.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, old)
        try:
            os.replace(stage, target)
        except Exception:
            if old is not None and old.exists() and not target.exists():
                os.replace(old, target)
            raise
        if signature(target) != verified_signature:
            raise RuntimeError('NAS destination changed after publication; originals retained')
        return (str(old) if old is not None else None), verified_signature

    def deletion_event(self, event):
        with self.journal_lock:
            with (self.run_dir / 'deletions.jsonl').open('a') as handle:
                handle.write(json.dumps(event) + '\n')
                handle.flush()
                os.fsync(handle.fileno())

    def delete_verified_local(self, details, nas_signature):
        """nas_signature is the unchanged file just hashed by publish/resume, never a persisted assertion."""
        if self.keep_raw or details.get('recovered_needs_review'):
            return {'raw_retained': True, 'deleted_files': 0, 'deleted_bytes': 0}
        self.check_stop()
        raw, dest = Path(details['source']), Path(details['destination'])
        root = next((r for r in self.roots if r in raw.parents), None)
        if root is None or raw.parent.name != 'videos' or raw.parent.parent.name != 'raw':
            raise RuntimeError('Unexpected source path; deletion refused')
        expected = self.nas / raw.relative_to(root).parents[2] / 'videos' / raw.name
        if dest != expected or not details.get('verified_at'):
            raise RuntimeError('NAS episode mapping/unverified record; deletion refused')
        src, out = details['source_info'], details['encoded_info']
        if not (src['count'] == src['declared_frames'] == out['count'] > 0) or (src['width'], src['height']) != (out['width'], out['height']):
            raise RuntimeError('Frame/dimension checks missing; deletion refused')
        confined_file(raw, root)
        confined_file(dest, self.nas)
        with dest.open('rb') as handle:
            os.fsync(handle.fileno())
        directory = os.open(str(dest.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if signature(raw) != details['source_signature'] or signature(dest) != nas_signature:
            raise RuntimeError('Source/NAS changed before cleanup; deletion refused')
        processed = raw.parents[2] / 'videos' / raw.name
        candidates = []
        if processed.exists() or processed.is_symlink():
            confined_file(processed, root)
            if processed.stat().st_mtime > datetime.fromisoformat(details['verified_at']).timestamp() + 2:
                raise RuntimeError('Local processing copy is newer than verified upload; deletion refused')
            candidates.append((processed, signature(processed), 'old_processed_copy'))
        candidates.append((raw, details['source_signature'], 'verified_raw'))
        for path, sig, kind in candidates:
            if has_open_writer(path):
                raise RuntimeError('Local video has an open writer; deletion deferred')
        deleted = []
        for path, sig, kind in candidates:
            self.check_stop()
            confined_file(path, root)
            confined_file(dest, self.nas)
            if signature(path) != sig or signature(dest) != nas_signature:
                raise RuntimeError('Local/NAS video changed before deletion')
            event = dict(action='delete_intent', path=str(path), signature=sig, bytes=sig[2], kind=kind,
                         nas=str(dest), nas_sha256=details['sha256'], nas_signature=nas_signature, time=time.time())
            self.deletion_event(event)
            # Repeat after the durable intent; do not remove files changed/opened during journaling.
            self.check_stop()
            confined_file(path, root)
            confined_file(dest, self.nas)
            if signature(path) != sig or signature(dest) != nas_signature or has_open_writer(path):
                raise RuntimeError('Files changed/opened after deletion intent; deletion refused')
            path.unlink()
            self.deletion_event(dict(event, action='deleted', time=time.time()))
            deleted.append(event)
        return {'raw_retained': False, 'deleted_files': len(deleted),
                'deleted_bytes': sum(e['bytes'] for e in deleted), 'deleted_at': datetime.now().isoformat()}

    def finish_verified(self, key, details, record, verified_record, nas_signature):
        recovered = details.get('recovered_needs_review', False)
        try:
            cleanup = self.delete_verified_local(details, nas_signature)
        except InterruptedError:
            raise
        except Exception as exc:
            cleanup = {'raw_retained': Path(details['source']).exists(), 'cleanup_error': str(exc)}
        details.pop('cleanup_error', None)
        details.update(cleanup)
        target = Path(details['destination'])
        write_json(record, details)
        write_json(verified_record, details)
        write_json(target.with_name(target.name + '.safe-upload.json'), details)
        message = ('RECOVERY verified; REVIEW REQUIRED; RAW RETAINED' if recovered else
                   'NAS checksum/frame validation OK; ' + ('RAW RETAINED' if details['raw_retained'] else 'LOCAL VIDEOS DELETED'))
        if cleanup.get('cleanup_error'):
            message += '; CLEANUP DEFERRED: ' + cleanup['cleanup_error']
        self.update(key, status='failed' if recovered else 'completed', progress=100,
                    current_frame=details['encoded_info']['count'], recovered_needs_review=recovered,
                    destination=str(target), message=message, **cleanup)

    def process_one(self, key, source, root):
        job_id = hashlib.sha256(key.encode()).hexdigest()
        record = self.run_dir / (job_id + '.json')
        verified_record = self.state / 'verified' / (job_id + '.json')
        try:
            self.check_stop()
            before = signature(source)
            if time.time() - source.stat().st_mtime < 60:
                raise RuntimeError('Recently modified raw file; defer until recording is closed')
            if verified_record.exists():
                saved = json.loads(verified_record.read_text())
                dest = Path(saved['destination'])
                if saved['source_signature'] == before and dest.exists():
                    nas_signature = signature(dest)
                    matches = file_hash(dest) == saved['sha256'] and signature(dest) == nas_signature
                    if matches:
                        self.finish_verified(key, saved, record, verified_record, nas_signature)
                        return
            self.update(key, status='processing', progress=1, message='Counting source packets; RAW RETAINED')
            raw = probe(source)
            if signature(source) != before:
                raise RuntimeError('Source changed during probe; original retained')
            relative = source.relative_to(root)
            episode = self.nas / relative.parents[2]
            reference = reference_timestamps(episode)
            self.update(key, total_frames=raw['count'], message='Encoding on captures1; RAW RETAINED', progress=5)
            with self.space(max(before[2] * 2, GIB)):
                work = self.scratch / self.run
                work.mkdir(parents=True, exist_ok=True)
                output = work / (job_id + '.avi')
                strict_log = self.run_dir / (job_id + '.ffmpeg.log')
                returncode = self.encode(source, output, strict_log, strict=True)
                recovered = returncode != 0
                if recovered:
                    self.update(key, message='Strict decode failed; recovering readable frames; RAW RETAINED')
                    output = work / (job_id + '.recovered.avi')
                    result = self.encode(source, output, self.run_dir / (job_id + '.recovery.log'), strict=False)
                    if result:
                        raise RuntimeError('Source cannot be fully decoded; raw and recovery files retained')
                self.check_stop()
                self.update(key, progress=65, message='Decoding every encoded frame for validation')
                encoded = probe(output, decode=True)
                if (encoded['width'], encoded['height']) != (raw['width'], raw['height']):
                    raise RuntimeError('Encoded dimensions mismatch')
                if encoded['count'] != raw['count']:
                    recovered = True
                if raw['declared_frames'] != raw['count']:
                    recovered = True
                warnings = []
                if reference and not reference.get('error'):
                    if reference['missing_ids'] or reference['non_increasing']:
                        warnings.append('Timestamp ID gaps/non-increasing IDs: ' + json.dumps(reference))
                    if reference['first_id'] == 1 and encoded['count'] < reference['span'] - 1:
                        recovered = True
                        warnings.append('Video contains fewer frames than timestamp trigger span')
                    elif encoded['count'] != reference['count']:
                        warnings.append('Video frame count differs from timestamp sample count')
                if signature(source) != before:
                    raise RuntimeError('Raw input changed during encoding; original retained')
                target = episode / ('recovery/videos' if recovered else 'videos') / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                nas_space = os.statvfs(str(target.parent))
                if nas_space.f_bavail * nas_space.f_frsize < output.stat().st_size + self.reserve_bytes:
                    raise RuntimeError('NAS space insufficient; originals retained')
                checksum = file_hash(output)
                stage = target.with_name('.' + target.name + '.' + self.run + '.uploading')
                details = {'source': str(source), 'source_signature': before, 'source_info': raw,
                           'encoded_info': encoded, 'timestamp_reference': reference, 'warnings': warnings,
                           'recovered_needs_review': recovered, 'destination': str(target),
                           'sha256': checksum, 'raw_retained': True, 'run': self.run}
                write_json(record, details)
                self.update(key, status='uploading', progress=80,
                            message='Copying to hidden NAS file, then SHA-256 verification; RAW RETAINED')
                details['previous_destination_backup'], nas_signature = self.publish_file(output, stage, target, checksum)
                details['verified_at'] = datetime.now().isoformat()
                write_json(target.with_name(target.name + '.safe-upload.json'), details)
                write_json(record, details)
                write_json(verified_record, details)
                # Only our generated scratch output is removed, after NAS byte verification.
                output.unlink()
                self.finish_verified(key, details, record, verified_record, nas_signature)
        except InterruptedError as exc:
            self.update(key, status='paused', message=str(exc))
        except Exception as exc:
            self.update(key, status='failed', message=str(exc) + '; RAW RETAINED')
            write_json(record, {'source': str(source), 'error': str(exc), 'raw_retained': True})

    def discover(self):
        jobs = []
        destinations = {}
        for root in self.roots:
            for path in sorted(root.glob('**/raw/videos/*.avi')):
                if path.is_symlink() or not path.is_file():
                    continue
                path.resolve().relative_to(root.resolve())
                key = str(path.relative_to(self.home))
                relative = path.relative_to(root)
                target_key = str(relative.parents[2] / path.name)
                self.jobs[key] = {'video_path': str(path), 'status': 'pending', 'progress': 0,
                                  'current_frame': 0, 'total_frames': 0, 'message': 'Queued; RAW RETAINED'}
                destinations.setdefault(target_key, []).append(key)
                jobs.append((key, path, root))
        duplicates = {key for group in destinations.values() if len(group) > 1 for key in group}
        for key in duplicates:
            self.update(key, status='failed', message='Duplicate episode/serial across roots; manual review; RAW RETAINED')
        # Small completed recordings give early verification results and keep
        # scratch reservations small. Only verified immutable local copies are removed.
        return sorted((job for job in jobs if job[0] not in duplicates), key=lambda job: job[1].stat().st_size)

    def run_all(self):
        if self.publish:
            import zmq
            context = zmq.Context()
            self.publisher = context.socket(zmq.PUB)
            self.publisher.setsockopt(zmq.LINGER, 0)
            self.publisher.bind('tcp://*:1234')
        try:
            jobs = self.discover()
            print('SAFE_UPLOAD_RUN=' + str(self.run_dir), flush=True)
            print('Queued {} raw videos; {}'.format(len(jobs), 'RAW RETAINED (--keep-raw)' if self.keep_raw else
                  'verified normal uploads DELETE LOCAL VIDEOS; failed/recovery inputs retained'), flush=True)
            self.save()
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self.process_one, *job) for job in jobs]
                while not all(future.done() for future in futures):
                    self.save()
                    time.sleep(2)
                for future in futures:
                    future.result()
            self.save()
            print(json.dumps(self.snapshot()['summary']), flush=True)
            if self.publisher:
                for _ in range(3):
                    self.save()
                    time.sleep(1)
        finally:
            if self.publisher:
                self.publisher.close()
                context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no-publish', action='store_true')
    parser.add_argument('--keep-raw', action='store_true', help='Retain local videos after successful NAS verification')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be in 1..4')
    state = Path.home() / '.local/state/paradex/safe-upload'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'uploader.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another safe uploader is already running')
        uploader = SafeUploader(state=state, workers=args.workers, publish=not args.no_publish, keep_raw=args.keep_raw)
        def stop(_signum, _frame):
            uploader.stop.set()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, stop)
        uploader.run_all()


if __name__ == '__main__':
    main()
