"""Read-only MANUS health monitor: no robot, command publisher or recording."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paradex.io.teleop.vive.receiver import ViveManusROSReceiver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=0, help='0: watch until Ctrl-C')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    receiver = ViveManusROSReceiver(use_vive=False, hand_side='bimanual', require_left_control=False)
    previous = {'Left': 0, 'Right': 0}
    started = last = time.monotonic()
    try:
        while not args.seconds or time.monotonic()-started < args.seconds:
            time.sleep(1)
            now = time.monotonic()
            d = receiver.get_diagnostics()
            for side in ('Left', 'Right'):
                v = d[side]
                v['receive_hz'] = round((v['valid_messages']-previous[side]) / (now-last), 1)
                previous[side] = v['valid_messages']
                v['status'] = 'OK' if v['is_fresh'] else ('STALE' if v['last_age_s'] is not None else 'NO DATA')
            last = now
            d['publishers'] = {t: receiver.node.count_publishers(t) for t in receiver.manus_topics}
            d['receiver_error'] = receiver.error_event.is_set()
            if args.json:
                print(json.dumps(d), flush=True)
            else:
                parts = []
                for side in ('Right', 'Left'):
                    v=d[side]; age='never' if v['last_age_s'] is None else '%.0f ms' % (1000*v['last_age_s'])
                    parts.append('%s: %s | %.1f Hz | age %s | valid %d invalid %d | glove %s | %s' %
                                 (side,v['status'],v['receive_hz'],age,v['valid_messages'],v['invalid_messages'],v['glove_id'],v['last_error'] or '-'))
                print('\n'.join(parts) + '\n  publishers: ' + str(d['publishers']), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.end()

if __name__ == '__main__':
    main()
