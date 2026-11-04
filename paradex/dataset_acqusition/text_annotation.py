"""One terminal input reader for capture keys and per-episode text annotations."""
import json
import os
import select
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
import uuid


class CaptureKeyboard:
    def __init__(self, events, input_fn=input, output=print):
        self.events, self.input_fn, self.output = events, input_fn, output
        self.lock = threading.Lock()
        self.closed = False
        self.pending = None
        self._stop_event = threading.Event()
        self._thread = None
        self._stdin_buffer = b''
        if input_fn is input:
            # input() holds stdin's buffered-reader lock while waiting. A
            # daemon doing that can abort Python during interpreter shutdown.
            self.input_fn = self._read_stdin_line

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name='capture-keyboard')
        self._thread.start()
        return self

    def _decode_stdin_line(self, line):
        try:
            # Surrogate-escaped input cannot be written as UTF-8 JSON either.
            return line.decode(sys.stdin.encoding or 'utf-8', errors='strict')
        except UnicodeDecodeError:
            self.output('Invalid terminal input encoding; line ignored. Please type it again.')
            return None

    def _read_stdin_line(self):
        fd = sys.stdin.fileno()
        while not self._stop_event.is_set():
            if b'\n' in self._stdin_buffer:
                line, self._stdin_buffer = self._stdin_buffer.split(b'\n', 1)
                text = self._decode_stdin_line(line)
                if text is not None:
                    return text
                continue
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                if self._stdin_buffer:
                    line, self._stdin_buffer = self._stdin_buffer, b''
                    text = self._decode_stdin_line(line)
                    if text is not None:
                        return text
                raise EOFError
            self._stdin_buffer += chunk
        raise EOFError

    def close(self):
        self._stop_event.set()
        with self.lock:
            self.closed = True
            if self.pending is not None:
                self.pending['done'].set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)

    def _run(self):
        try:
            while not self._stop_event.is_set():
                try:
                    line = self.input_fn()
                except UnicodeDecodeError:
                    self.output('Invalid terminal input encoding; line ignored. Please type it again.')
                    continue
                with self.lock:
                    if self.closed:
                        return
                    if self.pending is not None:
                        if self.pending['done'].is_set():
                            continue
                        self.pending['text'] = line.strip()
                        self.pending['done'].set()
                    else:
                        key = line.strip().lower()
                        if key in self.events:
                            self.events[key].set()
                            self.output('[%s] event triggered' % key)
        except (EOFError, KeyboardInterrupt):
            with self.lock:
                self.closed = True
                if self.pending is not None:
                    self.pending['done'].set()

    def annotate(self, episode_path, episode, loop_callback=None):
        request = {'text': None, 'done': threading.Event()}
        with self.lock:
            if self.closed:
                self.output('Annotation skipped: terminal input is closed.')
                return None
            self.pending = request
        self.output('Episode %s text annotation (Enter=skip; c/s/q are text here):' % episode)
        try:
            while not request['done'].wait(0.05):
                if loop_callback is not None:
                    loop_callback()
            text = request['text']
            if not text:
                self.output('Annotation skipped.')
                return None
            path = save_annotation(episode_path, episode, text)
            self.output('Annotation saved: ' + str(path))
            return path
        finally:
            with self.lock:
                self.pending = None


def save_annotation(episode_path, episode, text):
    folder = Path(episode_path)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'text_annotation.json'
    # An existing annotation belongs to this episode and must not be silently overwritten.
    if path.exists():
        path = folder / ('text_annotation_%s.json' % uuid.uuid4().hex[:8])
    temp = folder / ('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as handle:
        json.dump({'episode': episode, 'text': text,
                   'created_at': datetime.now(timezone.utc).isoformat()}, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path
