# Inspire bimanual wiring

`src/dataset_acquisition/hri/capture_robot.py` keeps logical hand sides separate
from physical Ethernet ports. A left hand connected to the right-hand port
still uses the left MANUS glove, left retargeting, left force limits, and
`hand_left` recording directory.

Set per-machine endpoints in `system/current/network.json`, alongside the
existing `inspire_ip.param` entry:

```json
"inspire_ip": {
    "name": "inspire",
    "left": {"ip": "192.168.11.210", "interface": "enp8s0f1"},
    "param": {"ip": "192.168.11.210", "port": 6000}
}
```

The optional `right` entry has the same format. Missing fields retain the
original layout: left `192.168.11.210` via `enp8s0f3`, right `192.168.12.210`
via `enp8s0f1`. Explicit `--inspire-left-interface`, `--inspire-left-ip`, and
the corresponding right-hand options override the machine settings.
The `param` entry continues to supply the TCP port and unimanual defaults.

On this PC, `enp8s0f1` has address `192.168.12.4/24`; the local address
`192.168.11.4/24` is assigned to `enp8s0f3`. Moving the left hand's cable
does not change its IP. A host route is also required so packets to the
left hand use its new port and a source in its subnet:

```bash
sudo ip route replace 192.168.11.210/32 dev enp8s0f1 src 192.168.11.4
ip route get 192.168.11.210
```

The route output should contain `dev enp8s0f1 src 192.168.11.4`. This route
only redirects that hand's address; it does not change the default gateway
or the rest of `192.168.11.0/24`. It is temporary and must be reapplied after
a reboot or network reconfiguration. It assumes the existing local
`192.168.11.4` address remains assigned to this PC.

With a left Inspire hand and a right Allegro V5, launch using:

```bash
python src/dataset_acquisition/hri/capture_robot.py \
    --name SESSION_NAME --hand-side bimanual \
    --hand-left inspire --hand-right allegro_v5
```

If teleoperation takes too long to start, add `--profile-startup` to see
which constructor is waiting and its elapsed time. Before the VIVE/MANUS
receiver starts, the default path connects the timestamp camera, initializes
both arms (including ROS service readiness), and connects the hands.
The remote camera controller connects asynchronously, so its constructor time
does not measure full remote camera readiness.

To measure startup delays while preserving camera initialization, use:

```bash
python src/dataset_acquisition/hri/capture_robot.py \
    --name SESSION_NAME --hand-side bimanual \
    --hand-left inspire_dftp --hand-right allegro_v5 \
    --profile-startup
```

Camera, trigger, and timestamp monitor construction remains at startup,
before arm, hand, and teleop receiver construction. Camera readiness is
still checked before timestamps and trigger output start when recording.
Timestamp camera connection exceptions now reach the
caller instead of leaving the constructor waiting forever; a backend call
that itself never returns is not covered by this error handling.

For two Inspire hands, use `--hand inspire`. For an absent right hand, use
`--hand-right none`. Confirm the startup message reports
`left=192.168.11.210 via enp8s0f1` before enabling teleoperation with the pedal.

For left-only operation, the same machine endpoint settings apply:

```bash
python src/dataset_acquisition/hri/capture_robot.py \
    --name SESSION_NAME --hand-side left --hand inspire --arm none --no-vive --pedal
```

This uses only the left MANUS glove; VIVE trackers and a right glove are
not required. The left glove controls
the robot hand and is not reused as the pause gesture. `--pedal` enables
teleoperation while held. The same host route above is required with this wiring.

For default MANUS retargeting, the left thumb uses separate anatomical inputs:
`ThumbDIPStretch` controls curl (raw channel 4), and `ThumbMCPSpread` controls
inward opposition (raw channel 5). Full curl is reached at 45 degrees and full
inward opposition at 35 degrees; both map to raw 0. This lets the thumb keep
curling while opposed into the palm, instead of driving both movements from
the old spread/stretch mapping. If DIP stretch is absent, PIP stretch supplies
curl. Older payloads missing both retain the legacy thumb mapping.

This applies to `inspire` and `inspire_dftp` in left-only and bimanual operation.
The other four fingers and the right hand retain their existing mapping.
Commands remain within 0–1000. The two full-travel angles are initial tuning
values in `paradex/retargetor/hand_regargetor.py` and still need physical
tracking validation on the hand.

When moving the left cable back, change its configured interface to
`enp8s0f3` (or remove the `left` entry) and remove the host route:

```bash
sudo ip route del 192.168.11.210/32 dev enp8s0f1
```
