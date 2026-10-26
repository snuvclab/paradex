"""Second stereo_track.py patch: `--policy-sub ENDPOINT` draws the policy runner's state (robustdex_deploy.viewer_overlay)
into the tracker's Viser scene each frame. Requires patch_stereo_track.py (cloud_pub) applied first. Idempotent.
Usage: patch_stereo_track_v2.py PATH/TO/stereo_track.py"""
import sys

p = sys.argv[1]
s = open(p).read()
assert 'cloud_pub' in s, 'apply patch_stereo_track.py first'
if 'policy_overlay' in s:
    print('already patched (v2)'); sys.exit(0)
edits = [
    ('    parser.add_argument("--cloud-pub", default=None,',
     '    parser.add_argument("--policy-sub", default=None,\n'
     '                        help="Draw the robustdex_deploy runner state (ZMQ SUB endpoint, e.g. tcp://127.0.0.1:5572) in the Viser scene")\n'
     '    parser.add_argument("--cloud-pub", default=None,'),
    ('        resources["cloud_pub"] = cloud_pub\n',
     '        resources["cloud_pub"] = cloud_pub\n'
     '    policy_overlay = None\n'
     '    if getattr(args, "policy_sub", None) and resources.get("server") is not None:\n'
     '        from robustdex_deploy.viewer_overlay import PolicyOverlay\n'
     '        policy_overlay = PolicyOverlay(resources["server"], args.policy_sub)\n'
     '        resources["policy_overlay"] = policy_overlay\n'),
    ('            if cloud_pub is not None:\n                cloud_pub.observe(tracked, display_rgb, record)\n',
     '            if cloud_pub is not None:\n                cloud_pub.observe(tracked, display_rgb, record)\n'
     '            if policy_overlay is not None:\n'
     '                try:\n                    policy_overlay.update()\n'
     '                except Exception as exc:  # never let the overlay break tracking\n'
     '                    print(f"[robustdex] overlay error: {exc!r}", flush=True)\n'),
    ('        if resources.get("cloud_pub") is not None:\n            resources["cloud_pub"].close()\n',
     '        if resources.get("cloud_pub") is not None:\n            resources["cloud_pub"].close()\n'
     '        if resources.get("policy_overlay") is not None:\n            resources["policy_overlay"].close()\n'),
]
for old, new in edits:
    assert s.count(old) == 1, ('anchor not unique/found', old[:70], s.count(old))
    s = s.replace(old, new)
open(p, 'w').write(s)
print('patched v2', p)
