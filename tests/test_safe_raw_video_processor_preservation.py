import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'paradex/video/safe_raw_video_processor.py'
spec = importlib.util.spec_from_file_location('safe_upload_module', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def setup(tmp_path):
    home = tmp_path / 'capture12'
    root = home / 'captures2'
    raw = root / 'capture/session/0/raw/videos/123.avi'
    raw.parent.mkdir(parents=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=10',
                    '-frames:v', '8', '-c:v', 'mjpeg', '-threads', '1', str(raw)], check=True)
    os.utime(raw, (time.time() - 120, time.time() - 120))
    uploader = module.SafeUploader(home=home, roots=[root], publish=False, reserve_bytes=0, keep_raw=True)
    jobs = uploader.discover()
    return uploader, jobs[0], raw, raw.read_bytes()


def test_success_verifies_all_frames_and_nas_checksum_without_deleting_raw(setup):
    uploader, job, raw, original = setup
    uploader.process_one(*job)
    info = uploader.jobs[job[0]]
    assert info['status'] == 'completed', info
    destination = Path(info['destination'])
    record = json.loads(destination.with_name(destination.name + '.safe-upload.json').read_text())
    assert record['encoded_info']['count'] == record['source_info']['count'] == 8
    assert module.file_hash(destination) == record['sha256']
    assert raw.read_bytes() == original
    assert not list(uploader.scratch.rglob('*.avi'))


def test_transfer_failure_preserves_original_and_existing_nas_file(setup, monkeypatch):
    uploader, job, raw, original = setup
    destination = uploader.nas / 'capture/session/0/videos/123.avi'
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'old NAS data')
    real_run = module.subprocess.run
    def fail_rsync(command, **kwargs):
        if command[0] == 'rsync':
            return subprocess.CompletedProcess(command, 23, '', 'injected transfer failure')
        return real_run(command, **kwargs)
    monkeypatch.setattr(module.subprocess, 'run', fail_rsync)
    uploader.process_one(*job)
    assert uploader.jobs[job[0]]['status'] == 'failed'
    assert raw.read_bytes() == original
    assert destination.read_bytes() == b'old NAS data'
    assert list(uploader.scratch.rglob('*.avi'))


def test_bad_nas_checksum_never_replaces_existing_destination(setup, monkeypatch):
    uploader, job, raw, original = setup
    destination = uploader.nas / 'capture/session/0/videos/123.avi'
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'keep me')
    real_hash = module.file_hash
    monkeypatch.setattr(module, 'file_hash', lambda p: 'bad hash' if p.name.endswith('.uploading') else real_hash(p))
    uploader.process_one(*job)
    assert 'checksum mismatch' in uploader.jobs[job[0]]['message']
    assert raw.read_bytes() == original
    assert destination.read_bytes() == b'keep me'


def test_replacement_keeps_previous_nas_file_as_backup(setup):
    uploader, job, raw, original = setup
    destination = uploader.nas / 'capture/session/0/videos/123.avi'
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'previous partial upload')
    uploader.process_one(*job)
    assert uploader.jobs[job[0]]['status'] == 'completed'
    record = json.loads(destination.with_name(destination.name + '.safe-upload.json').read_text())
    assert Path(record['previous_destination_backup']).read_bytes() == b'previous partial upload'
    assert raw.read_bytes() == original


def test_unfinalized_input_is_uploaded_under_recovery_and_original_is_retained(setup, monkeypatch):
    uploader, job, raw, original = setup
    real_probe = module.probe
    def unfinalized(path, decode=False):
        result = real_probe(path, decode)
        if path == raw:
            result['declared_frames'] = None
        return result
    monkeypatch.setattr(module, 'probe', unfinalized)
    uploader.process_one(*job)
    info = uploader.jobs[job[0]]
    assert info['status'] == 'failed' and info['recovered_needs_review']
    assert '/recovery/videos/' in info['destination']
    assert Path(info['destination']).exists()
    assert not (uploader.nas / 'capture/session/0/videos/123.avi').exists()
    assert raw.read_bytes() == original
    # Resuming a recovered job must keep the review warning.
    uploader.process_one(*job)
    assert uploader.jobs[job[0]]['status'] == 'failed'


def test_frame_loss_in_encoding_never_publishes_to_normal_video_path(setup, monkeypatch):
    uploader, job, raw, original = setup
    real_encode = uploader.encode
    def shorten(source, output, log, strict):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-frames:v', '4',
                        '-c:v', 'mjpeg', '-threads', '1', str(output)], check=True)
        return 0
    monkeypatch.setattr(uploader, 'encode', shorten)
    uploader.process_one(*job)
    info = uploader.jobs[job[0]]
    assert info['recovered_needs_review']
    assert '/recovery/videos/' in info['destination']
    assert raw.read_bytes() == original


def test_paused_job_leaves_input_untouched(setup):
    uploader, job, raw, original = setup
    uploader.stop.set()
    uploader.process_one(*job)
    assert uploader.jobs[job[0]]['status'] == 'paused'
    assert raw.read_bytes() == original


def test_auto_delete_removes_only_verified_raw_and_old_processing_copy(setup):
    uploader, job, raw, original = setup
    uploader.keep_raw = False
    processed = raw.parents[2] / 'videos' / raw.name
    processed.parent.mkdir()
    processed.write_bytes(b'old processing copy')
    uploader.process_one(*job)
    info = uploader.jobs[job[0]]
    assert info['status'] == 'completed' and not info['raw_retained'], info
    assert not raw.exists() and not processed.exists()
    assert Path(info['destination']).is_file()
    assert info['deleted_files'] == 2 and info['deleted_bytes'] == len(original) + len(b'old processing copy')
    assert uploader.snapshot()['raw_deletion_enabled']
    events = [json.loads(l) for l in (uploader.run_dir / 'deletions.jsonl').read_text().splitlines()]
    assert [e['action'] for e in events] == ['delete_intent', 'deleted'] * 2
    record = json.loads(next((uploader.state / 'verified').glob('*.json')).read_text())
    assert record['raw_retained'] is False and record['deleted_files'] == 2


def test_resume_auto_deletes_previously_verified_input_without_reencoding(setup, monkeypatch):
    uploader, job, raw, original = setup
    uploader.process_one(*job)
    assert raw.exists()
    uploader.keep_raw = False
    monkeypatch.setattr(uploader, 'encode', lambda *a, **k: pytest.fail('Completed uploads must not re-encode'))
    uploader.process_one(*job)
    info = uploader.jobs[job[0]]
    assert info['status'] == 'completed' and not raw.exists(), info
    assert module.file_hash(Path(info['destination'])) == json.loads(next((uploader.state/'verified').glob('*.json')).read_text())['sha256']


@pytest.mark.parametrize('problem', ['rsync', 'checksum', 'recovery'])
def test_auto_delete_preserves_inputs_on_transfer_checksum_or_recovery(setup, monkeypatch, problem):
    uploader, job, raw, original = setup
    uploader.keep_raw = False
    if problem == 'rsync':
        real_run = module.subprocess.run
        def fail(command, **kw):
            return subprocess.CompletedProcess(command, 23, '', 'failed') if command[0] == 'rsync' else real_run(command, **kw)
        monkeypatch.setattr(module.subprocess, 'run', fail)
    elif problem == 'checksum':
        real_hash = module.file_hash
        monkeypatch.setattr(module, 'file_hash', lambda p: 'bad' if p.name.endswith('.uploading') else real_hash(p))
    else:
        real_probe = module.probe
        def damaged(path, decode=False):
            r = real_probe(path, decode)
            if path == raw: r['declared_frames'] = None
            return r
        monkeypatch.setattr(module, 'probe', damaged)
    uploader.process_one(*job)
    assert uploader.jobs[job[0]]['status'] == 'failed'
    assert raw.read_bytes() == original


def test_auto_delete_protects_an_open_writer(setup):
    uploader, job, raw, original = setup
    uploader.keep_raw = False
    with raw.open('ab'):
        uploader.process_one(*job)
    info = uploader.jobs[job[0]]
    assert info['status'] == 'completed' and 'open writer' in info['cleanup_error'], info
    assert raw.read_bytes() == original


@pytest.mark.parametrize('change', ['raw', 'nas', 'stop'])
def test_files_changed_or_stop_after_durable_intent_are_protected(setup, monkeypatch, change):
    uploader, job, raw, original = setup
    uploader.process_one(*job)
    uploader.keep_raw = False
    dest = Path(uploader.jobs[job[0]]['destination'])
    journal = uploader.deletion_event
    def race(event):
        journal(event)
        if event['action'] == 'delete_intent':
            if change == 'stop': uploader.stop.set()
            else:
                p = raw if change == 'raw' else dest
                with p.open('ab') as handle: handle.write(b'concurrent change')
    monkeypatch.setattr(uploader, 'deletion_event', race)
    uploader.process_one(*job)
    assert raw.exists()
    if change != 'raw': assert raw.read_bytes() == original
    assert uploader.jobs[job[0]]['status'] == ('paused' if change == 'stop' else 'completed')


def test_auto_delete_refuses_wrong_episode_mapping_and_new_processing_copy(setup):
    uploader, job, raw, original = setup
    uploader.process_one(*job)
    record_path = next((uploader.state / 'verified').glob('*.json'))
    record = json.loads(record_path.read_text())
    dest = Path(record['destination'])
    uploader.keep_raw = False
    wrong = uploader.nas / 'another_episode/videos/123.avi'
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(dest.read_bytes())
    record['destination'] = str(wrong)
    record_path.write_text(json.dumps(record))
    uploader.process_one(*job)
    assert raw.exists() and 'mapping' in uploader.jobs[job[0]]['cleanup_error']
    record['destination'] = str(dest)
    record_path.write_text(json.dumps(record))
    processed = raw.parents[2] / 'videos' / raw.name
    processed.parent.mkdir()
    processed.write_bytes(b'new data')
    os.utime(processed, (time.time()+10, time.time()+10))
    uploader.process_one(*job)
    assert raw.exists() and processed.exists()
    assert 'newer' in uploader.jobs[job[0]]['cleanup_error']
