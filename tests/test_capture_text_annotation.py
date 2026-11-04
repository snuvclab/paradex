import importlib.util
from pathlib import Path
import json
import queue
import subprocess
import sys
import threading
import time

import pytest

source=Path(__file__).resolve().parents[1]/'paradex/dataset_acqusition/text_annotation.py'
if not source.exists():source=Path(__file__).with_name('text_annotation.py')
spec=importlib.util.spec_from_file_location('annotations',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def wait(fn):
    for _ in range(100):
        if fn():return
        time.sleep(.01)
    assert fn()

def test_annotation_and_keys_share_single_input_reader(tmp_path):
    lines=queue.Queue();events={'c':threading.Event(),'q':threading.Event()}
    def read():
        value=lines.get()
        if value is None:raise EOFError
        return value
    k=m.CaptureKeyboard(events,input_fn=read,output=lambda s:None).start()
    lines.put('c');wait(events['c'].is_set)
    t=threading.Thread(target=k.annotate,args=(tmp_path,7));t.start();wait(lambda:k.pending is not None)
    lines.put('q');t.join(2);assert not t.is_alive()
    assert not events['q'].is_set()
    assert json.loads((tmp_path/'text_annotation.json').read_text())['text']=='q'
    lines.put('q');wait(events['q'].is_set);lines.put(None)

def test_unicode_and_existing_annotation_preserved(tmp_path):
    p=m.save_annotation(tmp_path,1,'큐브를 잡고 들어 올림')
    q=m.save_annotation(tmp_path,1,'두 번째 메모')
    assert p!=q and json.loads(p.read_text())['text']=='큐브를 잡고 들어 올림'
    assert json.loads(q.read_text())['episode']==1

def test_blank_and_closed_input_skip_without_creating_file(tmp_path):
    for line in ('',None):
        k=m.CaptureKeyboard({},output=lambda s:None)
        def finish():
            wait(lambda:k.pending is not None)
            with k.lock:
                k.pending['text']=line;k.pending['done'].set()
        t=threading.Thread(target=finish);t.start();assert k.annotate(tmp_path,2) is None;t.join()
    k.closed=True;assert k.annotate(tmp_path,3) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('close_keyboard', [False, True])
def test_exit_with_open_stdin_does_not_abort(close_keyboard):
    # Keep stdin open while the main thread exits, matching q/Ctrl-C or a
    # startup exception while the keyboard reader is waiting for a line.
    probe = '''
import importlib.util
import resource
import sys
import time
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
spec = importlib.util.spec_from_file_location('annotations', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
keyboard = module.CaptureKeyboard({}).start()
time.sleep(0.1)
if sys.argv[2] == 'True':
    keyboard.close()
    assert not keyboard._thread.is_alive()
'''
    child = subprocess.Popen(
        [sys.executable, '-c', probe, str(source), str(close_keyboard)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        code = child.wait(timeout=5)
        assert code == 0, child.stderr.read()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()

@pytest.mark.parametrize('errors', ['strict', 'surrogateescape'])
def test_bad_utf8_line_is_discarded_and_next_capture_key_works(monkeypatch, errors):
    import os
    from types import SimpleNamespace
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(fileno=lambda: read_fd, encoding='utf-8', errors=errors))
    event=threading.Event();messages=[]
    keyboard=m.CaptureKeyboard({'c':event},output=messages.append).start()
    try:
        os.write(write_fd,b'\xe3c\n')
        wait(lambda:any('encoding' in s for s in messages))
        assert not event.is_set() and keyboard._thread.is_alive()
        os.write(write_fd,b'c\n');wait(event.is_set)
        assert keyboard._thread.is_alive()
    finally:
        keyboard.close();os.close(read_fd);os.close(write_fd)


def test_split_utf8_chunks_preserve_korean_annotation(monkeypatch,tmp_path):
    import os
    from types import SimpleNamespace
    read_fd,write_fd=os.pipe()
    monkeypatch.setattr(m.sys,'stdin',SimpleNamespace(fileno=lambda:read_fd,encoding='utf-8',errors='strict'))
    keyboard=m.CaptureKeyboard({},output=lambda s:None).start()
    worker=threading.Thread(target=keyboard.annotate,args=(tmp_path,1));worker.start()
    try:
        wait(lambda:keyboard.pending is not None)
        os.write(write_fd,b'\xe3\n')
        time.sleep(.03)
        assert worker.is_alive() and not (tmp_path/'text_annotation.json').exists()
        raw='큐브를 잡음'.encode('utf-8')
        os.write(write_fd,raw[:1]);time.sleep(.03);os.write(write_fd,raw[1:]+b'\n')
        worker.join(2);assert not worker.is_alive()
        assert json.loads((tmp_path/'text_annotation.json').read_text())['text']=='큐브를 잡음'
    finally:
        keyboard.close();worker.join(2);os.close(read_fd);os.close(write_fd)


def test_invalid_partial_line_at_eof_closes_reader(monkeypatch):
    import os
    from types import SimpleNamespace
    read_fd,write_fd=os.pipe()
    monkeypatch.setattr(m.sys,'stdin',SimpleNamespace(fileno=lambda:read_fd,encoding='utf-8',errors='strict'))
    keyboard=m.CaptureKeyboard({},output=lambda s:None).start()
    try:
        os.write(write_fd,b'\xe3');os.close(write_fd)
        wait(lambda:keyboard.closed)
    finally:
        keyboard.close();os.close(read_fd)
