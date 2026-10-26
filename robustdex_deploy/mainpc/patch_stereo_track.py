"""Add a `--cloud-pub ENDPOINT` option to paradex src/capture/camera/stereo_track.py: every tracked frame, the object
model (robot base frame) + ICP transform are published over ZMQ (robustdex_deploy.cloud.CloudPublisher) for the policy
runner. Idempotent; writes stereo_track.py.bak_robustdex once. Usage: patch_stereo_track.py PATH/TO/stereo_track.py"""
import shutil
import sys

p = sys.argv[1]
s = open(p).read()
if 'cloud_pub' in s:
    print('already patched'); sys.exit(0)
shutil.copy(p, p + '.bak_robustdex')
edits = [
    ('    parser.add_argument("--grasp-extra-close", type=float, default=0.20,',
     '    parser.add_argument("--cloud-pub", default=None,\n'
     '                        help="Publish the tracked object model (robot frame) over ZMQ PUB for robustdex_deploy, e.g. tcp://127.0.0.1:5571")\n'
     '    parser.add_argument("--grasp-extra-close", type=float, default=0.20,'),
    ('    grasp_retrieval = None\n',
     '    grasp_retrieval = None\n'
     '    cloud_pub = None\n'
     '    if getattr(args, "cloud_pub", None):\n'
     '        from robustdex_deploy.cloud import CloudPublisher\n'
     '        cloud_pub = CloudPublisher(args.cloud_pub)\n'
     '        resources["cloud_pub"] = cloud_pub\n'
     '        print(f"[robustdex] publishing object cloud on {args.cloud_pub}", flush=True)\n'),
    ('                grasp_retrieval.observe(tracked, display_rgb, record)\n',
     '                grasp_retrieval.observe(tracked, display_rgb, record)\n'
     '            if cloud_pub is not None:\n'
     '                cloud_pub.observe(tracked, display_rgb, record)\n'),
    ('        if resources.get("grasp_retrieval") is not None:\n            resources["grasp_retrieval"].close()\n',
     '        if resources.get("grasp_retrieval") is not None:\n            resources["grasp_retrieval"].close()\n'
     '        if resources.get("cloud_pub") is not None:\n            resources["cloud_pub"].close()\n'),
]
for old, new in edits:
    assert s.count(old) == 1, ('anchor not unique/found', old[:60], s.count(old))
    s = s.replace(old, new)
open(p, 'w').write(s)
print('patched', p)
