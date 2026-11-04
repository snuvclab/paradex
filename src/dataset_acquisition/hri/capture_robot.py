import os
import sys
import argparse
import atexit
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import chime
# chime.theme('pokemon')

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index
from paradex.dataset_acqusition.text_annotation import CaptureKeyboard
from paradex.utils.system import get_pc_list, network_info
from paradex.io.robot_controller.inspire_f1_tactile_plotter import InspireF1RealtimeTactilePlotter
from paradex.io.streamdeck_pedal import MiddlePedalState

EXCLUDED_PCS = {}
camera_pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]
# Physical ports belong to the machine configuration, not the retargeting side.
# Keep the existing layout as the fallback for machines without side settings.
inspire_network = network_info.get('inspire_ip', {})
inspire_left_network = inspire_network.get('left', {})
inspire_right_network = inspire_network.get('right', {})

parser = argparse.ArgumentParser()

parser.add_argument('--device', choices=['xsens', 'occulus', 'vive'], default="vive")
parser.add_argument('--hand-side', choices=['right', 'left', 'bimanual'], default="right")
parser.add_argument(
    '--pedal', action='store_true',
    help='Use the middle Stream Deck pedal for unimanual arm/hand teleop: hold=enable, release=pause. Bimanual always uses the pedal.',
)
vive_group = parser.add_mutually_exclusive_group()
vive_group.add_argument(
    '--use-vive', dest='use_vive', action='store_true',
    help='Use the VIVE wrist tracker for arm/wrist pose fusion (default).',
)
vive_group.add_argument(
    '--no-vive', dest='use_vive', action='store_false',
    help='Use Manus glove data only; suitable for hand-only teleoperation.',
)
parser.set_defaults(use_vive=True)
camera_group = parser.add_mutually_exclusive_group()
camera_group.add_argument(
    '--camera',
    dest='camera_mode',
    nargs='?',
    const='capture',
    choices=['capture', 'preview'],
    default='capture',
    help=(
        "Enable remote camera recording. Use '--camera=preview' to also show "
        'the independent live capture-PC preview window.'
    ),
)
camera_group.add_argument(
    '--no-camera',
    dest='camera_mode',
    action='store_const',
    const='off',
    help='Disable remote cameras, sync generator, and timestamp monitor.',
)
camera_group.add_argument(
    '--camera-preview',
    dest='camera_mode',
    action='store_const',
    const='preview',
    help="Alias for '--camera=preview'.",
)
parser.add_argument('--camera-preview-port', type=int, default=5484)
parser.add_argument('--camera-preview-refresh-interval', type=float, default=0.2)
parser.add_argument('--camera-preview-request-timeout', type=float, default=1.5)
parser.add_argument(
    '--no-timestamp',
    dest='timestamp',
    action='store_false',
    help='Disable the timestamp monitor connection.',
)
parser.set_defaults(timestamp=True)
parser.add_argument(
    '--profile-startup', action='store_true',
    help='Print initialization stage names and elapsed seconds for cameras, arms, hands, and teleop.',
)
parser.add_argument('--arm', type=str, default="xarm",
                    help="Arm controller name. Use 'none' (or empty) to disable arm control.")
parser.add_argument('--hand', type=str, default="allegro_v5",
                    help="Hand controller/retargetor name. Use 'none' for arm-only teleop; 'allegro_v5' with --allegro-retarget direct|dexpilot|legacy-0818; 'wuji' for optimization, 'wuji_direct' for direct mapping, or 'wuji_hybrid' for opt thumb + direct fingers.")
parser.add_argument('--hand-left', '--hand_left', default=None,
                    help="Left hand controller/retargetor (bimanual only). Defaults to --hand; use 'none' to disable.")
parser.add_argument('--hand-right', '--hand_right', default=None,
                    help="Right hand controller/retargetor (bimanual only). Defaults to --hand; use 'none' to disable.")
parser.add_argument(
    '--allegro-retarget', choices=['direct', 'dexpilot', 'legacy-0818'], default='direct',
    help='Allegro V5 only: direct MANUS angles (default), dexpilot optimization (right hand), or legacy-0818 mapping from fe18720cf.',
)
parser.add_argument('--allegro-thumb-distal-mapping', choices=['sine', 'linear'], default='linear',
                    help='Allegro V5 direct only: linear maps MANUS DIP degrees 1:1 with joint limits (default); sine restores the previous thumb-tip curve.')
parser.add_argument('--capture_root', type=str, default="snapgrasp")
parser.add_argument('--name', type=str, required=True)
tactile_group = parser.add_mutually_exclusive_group()
tactile_group.add_argument('--tactile', dest='tactile', action='store_true', default=None,
                           help='Record hand tactile data (default: enabled for Allegro V5).')
tactile_group.add_argument('--no-tactile', dest='tactile', action='store_false',
                           help='Disable tactile recording.')
parser.add_argument('--ip', action="store_true")
parser.add_argument(
    '--allegro-command-rate-hz',
    type=float,
    default=30.0,
    help=(
        'Maximum live target-update rate for Allegro V5 hands. Defaults to '
        '30 Hz, matching the Allegro alignment experiment; the driver holds '
        'the latest target between updates. Use 0 to disable this capture '
        'target-update limit; the driver publishing rate is unchanged.'
    ),
)
parser.add_argument(
    '--allegro-teleop-log',
    nargs='?',
    const='auto',
    default=None,
    metavar='PATH',
    help=(
        'Record every commanded Allegro V5 tick with the MANUS frame/ergonomics, '
        'retargeter target, controller target, and measured ROS feedback. '
        'Without PATH, save a timestamped .npz below this capture name.'
    ),
)
parser.add_argument(
    '--inspire-right-interface',
    default=inspire_right_network.get('interface', 'enp8s0f1'),
    help='Network interface for the right Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--inspire-right-ip',
    default=inspire_right_network.get('ip', '192.168.12.210'),
    help='IP address for the right Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--inspire-left-interface',
    default=inspire_left_network.get('interface', 'enp8s0f3'),
    help='Network interface for the left Inspire Modbus TCP hand.',
)
parser.add_argument(
    '--inspire-left-ip',
    default=inspire_left_network.get('ip', '192.168.11.210'),
    help='IP address for the left Inspire Modbus TCP hand.',
)
def _inspire_force_limits(text):
    values = [int(value) for value in text.split(',')]
    if len(values) != 6 or not all(0 <= value <= 3000 for value in values):
        raise argparse.ArgumentTypeError('expected six comma-separated values in 0..3000')
    return values


parser.add_argument(
    '--inspire-left-force-limits',
    type=_inspire_force_limits,
    default=[400, 400, 400, 400, 400, 400],
    help='Left Inspire FORCE_SET (little,ring,middle,index,thumb_bend,thumb_rotate).',
)
parser.add_argument(
    '--inspire-right-force-limits',
    type=_inspire_force_limits,
    # The right hand's thumb bend is mechanically stiffer and stalls near
    # ANGLE 245 at 400; 600 reaches the end of travel.
    default=[400, 400, 400, 400, 600, 400],
    help='Right Inspire FORCE_SET (little,ring,middle,index,thumb_bend,thumb_rotate).',
)
parser.add_argument(
    '--inspire-retarget',
    choices=['manus', 'dexpilot'],
    default='manus',
    help=(
        "Inspire finger retargeting: 'manus' maps MANUS ergonomics angles "
        "(with the thumb lock); 'dexpilot' runs AnyTeleop-style fingertip "
        "vector optimization with a glove thumb-IP bend blend."
    ),
)
parser.add_argument(
    '--inspire-thumb-bend-weight',
    type=float,
    default=0.7,
    help='DexPilot thumb bend blend: 0=original DexPilot, 1=glove IP bend (default: 0.7).',
)
parser.add_argument(
    '--inspire-thumb-full-bend-deg',
    type=float,
    default=60.0,
    help='Glove thumb IP angle mapped to fully closed in DexPilot mode (default: 60 deg).',
)
parser.add_argument(
    '--visualize-tactile-realtime',
    action="store_true",
    help=(
        'Show live tactile feedback. Inspire F1 uses matplotlib; Allegro V5 '
        'uses a Viser feedback mesh with fingertip arrows.'
    ),
)
parser.add_argument(
    '--allegro-visualization-rate-hz',
    type=float,
    default=100.0,
    help='Viser refresh rate for lightweight Allegro tactile arrows.',
)
parser.add_argument(
    '--allegro-tactile-display-max',
    type=float,
    default=1000.0,
    help='Raw tactile magnitude represented by the maximum arrow length.',
)
parser.add_argument('--vive-arm-smoothing-ms', type=float, default=0,
                    help='VIVE arm position/rotation smoothing time constant in ms (default 80; 0 disables).')
parser.add_argument('--xarm-servo-api' , choices=["cartesian_aa", "angle_j"], default="cartesian_aa")
parser.add_argument('--teleop-control-mode', choices=['split', 'shared'], default='split',
                    help='Arm/hand target loops: independent threads and input freshness (default); shared keeps the previous loop.')
parser.add_argument('--teleop-receiver-mode', choices=['split', 'shared'], default='split',
                    help='VIVE/MANUS callbacks: split uses separate nodes/executor threads (default); shared uses one thread for comparison.')
parser.add_argument(
    '--scale', '--hand-scale',
    dest='hand_scale',
    type=float,
    default=1.15

    ,
    help="Keypoint scale for Allegro/Inspire DexPilot and Wuji. Default: 1.15.",
)
args = parser.parse_args()
if args.inspire_retarget == 'dexpilot':
    if not np.isfinite(args.inspire_thumb_bend_weight) or not 0 <= args.inspire_thumb_bend_weight <= 1:
        parser.error('--inspire-thumb-bend-weight must be finite and in [0, 1]')
    if not np.isfinite(args.inspire_thumb_full_bend_deg) or not 0 < args.inspire_thumb_full_bend_deg <= 180:
        parser.error('--inspire-thumb-full-bend-deg must be finite and in (0, 180]')


def _normalize_optional_name(name):
    if name is not None and name.strip().lower() in ("", "none", "null"):
        return None
    return name


args.arm = _normalize_optional_name(args.arm)
args.hand = _normalize_optional_name(args.hand)
if args.hand_side != 'bimanual' and (
    args.hand_left is not None or args.hand_right is not None
):
    parser.error('--hand-left and --hand-right require --hand-side bimanual.')
if args.hand_side == 'bimanual':
    args.hand_left = _normalize_optional_name(
        args.hand if args.hand_left is None else args.hand_left
    )
    args.hand_right = _normalize_optional_name(
        args.hand if args.hand_right is None else args.hand_right
    )
    # Resolve defaults here so an explicit 'none' cannot fall back to --hand.
    session_hand_kwargs = dict(hand=None, hand_left=args.hand_left, hand_right=args.hand_right)
    active_hands = {'left': args.hand_left, 'right': args.hand_right}
else:
    session_hand_kwargs = dict(hand=args.hand)
    active_hands = {args.hand_side: args.hand}
if args.tactile is None:
    args.tactile = 'allegro_v5' in active_hands.values()
if args.allegro_retarget == 'dexpilot' and (
    active_hands.get('right') != 'allegro_v5'
    or active_hands.get('left') == 'allegro_v5'
):
    parser.error('--allegro-retarget dexpilot requires a right Allegro V5 hand and no left Allegro V5 hand.')
if not np.isfinite(args.allegro_command_rate_hz) or args.allegro_command_rate_hz < 0.0:
    parser.error('--allegro-command-rate-hz must be finite and nonnegative (0 disables the limit).')
if args.camera_preview_port <= 0:
    parser.error('--camera-preview-port must be positive.')
if args.camera_preview_refresh_interval <= 0.0:
    parser.error('--camera-preview-refresh-interval must be positive.')
if args.camera_preview_request_timeout <= 0.0:
    parser.error('--camera-preview-request-timeout must be positive.')
if args.allegro_visualization_rate_hz <= 0.0:
    parser.error('--allegro-visualization-rate-hz must be positive.')
if args.allegro_tactile_display_max <= 0.0:
    parser.error('--allegro-tactile-display-max must be positive.')

if not np.isfinite(args.vive_arm_smoothing_ms) or not 0 <= args.vive_arm_smoothing_ms <= 1000:
    parser.error('--vive-arm-smoothing-ms must be finite and in [0, 1000]')

camera_enabled = args.camera_mode != 'off'
camera_preview_enabled = args.camera_mode == 'preview'
allegro_v5_hands = {'allegro_v5'}
allegro_realtime_visualization = (
    args.visualize_tactile_realtime
    and args.hand in allegro_v5_hands
    and args.hand_side == 'right'
)
allegro_teleop_diagnostic_path = None
if args.allegro_teleop_log is not None:
    if args.hand not in allegro_v5_hands or args.hand_side != 'right':
        parser.error('--allegro-teleop-log currently requires --hand allegro_v5 and --hand-side right.')
    if args.allegro_teleop_log == 'auto':
        diagnostic_name = datetime.now().strftime('allegro_teleop_%Y%m%d_%H%M%S.npz')
        allegro_teleop_diagnostic_path = (
            Path(shared_dir) / 'capture' / args.capture_root / args.name / diagnostic_name
        )
    else:
        allegro_teleop_diagnostic_path = Path(args.allegro_teleop_log).expanduser()


stop_event = Event()
save_event = Event()
exit_event = Event()
grasp_yes_event = Event()
grasp_no_event = Event()
events = {"save": save_event, "stop": stop_event, "exit": exit_event}

capture_keyboard = CaptureKeyboard(
    {
        "c": save_event,
        "q": exit_event,
        "s": stop_event,
        "y": grasp_yes_event,
        "n": grasp_no_event,
    }
)
capture_keyboard.start()
atexit.register(capture_keyboard.close)
print("Keyboard control: c=start capture, s=stop capture, q=exit, y=grasp success, n=grasp fail")
print("In keyboard_control mode, gesture states 2/3 do not control session start/stop/exit.")

unimanual_pedal = args.pedal and args.hand_side != "bimanual"
pedal_state = MiddlePedalState() if args.pedal or args.hand_side == "bimanual" else None
if pedal_state is not None:
    atexit.register(pedal_state.close)

inspire_sides = [
    side for side, hand in active_hands.items()
    if args.hand_side in ('left', 'bimanual') and hand in ('inspire', 'inspire_dftp')
]
hand_kwargs = None
if inspire_sides:
    inspire_hand_kwargs = {
        side: {
            "interface": getattr(args, f'inspire_{side}_interface'),
            "host": getattr(args, f'inspire_{side}_ip'),
            "force_limits": getattr(args, f'inspire_{side}_force_limits'),
        }
        for side in inspire_sides
    }
    for side, settings in inspire_hand_kwargs.items():
        print(
            f"Inspire Modbus TCP: {side}={settings['host']} "
            f"via {settings['interface']} force={settings['force_limits']}"
        )
    hand_kwargs = (
        inspire_hand_kwargs
        if args.hand_side == 'bimanual'
        else inspire_hand_kwargs[args.hand_side]
    )

try:
    cs = CaptureSession(
        camera=camera_enabled,
        realsense=False,
        arm=args.arm,
        **session_hand_kwargs,
        teleop=args.device,
        hand_side=args.hand_side,
        events=events,
        tactile=args.tactile or allegro_realtime_visualization,
        ip=args.ip or bool(inspire_sides),
        hand_kwargs=hand_kwargs,
        timestamp=args.timestamp,
        profile_startup=args.profile_startup,
        camera_pc_list=camera_pc_list,
        arm_kwargs={"servo_api": args.xarm_servo_api} if args.arm == "xarm" else None,
        hand_scale=args.hand_scale,
        inspire_retarget=args.inspire_retarget,
        inspire_thumb_bend_weight=args.inspire_thumb_bend_weight,
        inspire_thumb_full_bend_deg=args.inspire_thumb_full_bend_deg,
        allegro_retarget=args.allegro_retarget,
        allegro_thumb_distal_mapping=args.allegro_thumb_distal_mapping,
        hand_command_rate_hz=(
            args.allegro_command_rate_hz
            if 'allegro_v5' in active_hands.values() and args.allegro_command_rate_hz > 0.0
            else None
        ),
        allegro_teleop_diagnostic_path=allegro_teleop_diagnostic_path,
        use_vive=args.use_vive,
        teleop_receiver_mode=args.teleop_receiver_mode,
        teleop_control_mode=args.teleop_control_mode,
        vive_arm_smoothing_tau_s=args.vive_arm_smoothing_ms / 1000.0,
        require_left_control=(args.use_vive and not unimanual_pedal and args.hand_side != 'left'
                              and any(name is not None for name in active_hands.values())),
    )
except Exception:
    if pedal_state is not None:
        pedal_state.close()
    raise

if allegro_teleop_diagnostic_path is not None:
    print(f"Allegro teleop diagnostic will be saved to: {allegro_teleop_diagnostic_path}")

bimanual_state_provider = (
    pedal_state.get_state
    if pedal_state is not None and args.hand_side == "bimanual"
    else None
)
unimanual_state_provider = pedal_state.get_state if unimanual_pedal else None

tactile_plotter = None
if args.visualize_tactile_realtime:
    if args.hand_side == "bimanual":
        print("Realtime tactile visualization is not supported in bimanual mode. Ignoring option.")
    elif args.hand in allegro_v5_hands and args.hand_side != "right":
        print("Realtime Allegro visualization currently supports the right hand only. Ignoring option.")
    elif args.hand in allegro_v5_hands:
        from paradex.visualization.allegro_realtime import AllegroRealtimeViser

        tactile_plotter = AllegroRealtimeViser(
            cs.hand,
            update_rate_hz=args.allegro_visualization_rate_hz,
            tactile_display_max=args.allegro_tactile_display_max,
        )
        tactile_plotter.start()
    elif args.hand != "inspire_f1":
        print("Realtime tactile visualization supports inspire_f1 and Allegro V5. Ignoring option.")
    elif not args.tactile:
        print("Realtime tactile visualization requires --tactile. Ignoring option.")
    else:
        tactile_plotter = InspireF1RealtimeTactilePlotter(cs.hand)
        if tactile_plotter.enabled:
            tactile_plotter.start()

camera_preview = None
if camera_preview_enabled:
    from paradex.io.camera_system.capture_pc_preview import CapturePcPreviewGui

    camera_preview = CapturePcPreviewGui(
        pc_list=camera_pc_list,
        port=args.camera_preview_port,
        refresh_interval=args.camera_preview_refresh_interval,
        request_timeout=args.camera_preview_request_timeout,
        side_panel_provider=getattr(tactile_plotter, "render_bgr", None),
    )
    camera_preview.start()


def refresh_guis(_session=None):
    """Process preview GUI events on the teleoperation main thread."""

    if camera_preview is not None:
        camera_preview.show()


def wait_for_grasp_result():
    grasp_yes_event.clear()
    grasp_no_event.clear()
    print("Grasp success? Press y or n.")

    while not exit_event.is_set():
        refresh_guis()
        if grasp_yes_event.is_set():
            return True
        if grasp_no_event.is_set():
            return False
        time.sleep(0.01)

    return None


name = args.name

last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, args.name)))
success_count = 0
fail_count = 0

try:
    while not exit_event.is_set():
        state = cs.teleop(
            session_events=events,
            state_policy="keyboard_control",
            loop_callback=refresh_guis,
            bimanual_state_provider=bimanual_state_provider,
            unimanual_state_provider=unimanual_state_provider,
        )

        if state == "exit":
            break

        if state != "start":
            continue

        last_idx += 1
        print("Prepare to record new session:", name, "episode:", last_idx)
        episode_rel_path = os.path.join("capture", args.capture_root, args.name, str(last_idx))
        episode_abs_path = os.path.join(shared_dir, episode_rel_path)
        # Match capture_hand.py: an idle-time 's' must not immediately stop
        # the next capture after 'c' is pressed.
        stop_event.clear()
        cs.start(episode_rel_path)
        # chime.info(sync=True)
        print("Starting new recording session:", name)
        print("Capturing index:", last_idx)

        state = cs.teleop(
            session_events=events,
            state_policy="keyboard_control",
            loop_callback=refresh_guis,
            bimanual_state_provider=bimanual_state_provider,
            unimanual_state_provider=unimanual_state_provider,
        )
        print("Stopping recording session:", name)
        cs.stop()
        print("Stopped recording session:", name)
        try:
            from paradex.dataset_acqusition.capture_integrity import (
                format_integrity_report,
                validate_immediate_capture,
            )

            integrity_report = validate_immediate_capture(
                episode_abs_path,
                arm_enabled=args.arm is not None,
                hand_enabled=any(hand is not None for hand in active_hands.values()),
                hand_sides=[side for side, hand in active_hands.items() if hand is not None],
                bimanual=args.hand_side == "bimanual",
                teleop_enabled=args.device is not None,
                camera_enabled=args.camera_mode != "off",
                timestamp_expected=(
                    args.camera_mode != "off" and args.timestamp
                ),
            )
            print(format_integrity_report(integrity_report))
        except Exception as exc:
            print(
                f"Immediate integrity check could not run: {exc}",
                file=sys.stderr,
            )

        capture_keyboard.annotate(
            episode_abs_path, last_idx, loop_callback=refresh_guis,
        )

        save_event.clear()
        stop_event.clear()

        # if state != "exit":
        #     grasp_success = wait_for_grasp_result()
        #     if grasp_success is not None:
        #         success_count += int(grasp_success)
        #         fail_count += int(not grasp_success)
        #         os.makedirs(episode_abs_path, exist_ok=True)
        #         grasp_json_path = os.path.join(episode_abs_path, "grasp_result.json")
        #         with open(grasp_json_path, "w") as file:
        #             json.dump(
        #                 {
        #                     "episode": last_idx,
        #                     "grasp_success": grasp_success,
        #                 },
        #                 file,
        #                 indent=2,
        #             )

        #         # paired_human_episode = int(
        #         #     input(
        #         #         "Enter the episode number of paired human sequence "
        #         #         f"for {args.name}: "
        #         #     )
        #         # )
        #         # paired_info_json_path = os.path.join(
        #         #     episode_abs_path,
        #         #     "paired_human_episode.json",
        #         # )
        #         # with open(paired_info_json_path, "w") as file:
        #         #     json.dump(
        #         #         {
        #         #             "human hand episode": last_idx,
        #         #             "paired human episode": paired_human_episode,
        #         #         },
        #         #         file,
        #         #         indent=2,
        #         #     )

        #         print(
        #             f"Current Success count: {success_count} / "
        #             f"Failure count: {fail_count}"
        #         )

        # grasp_yes_event.clear()
        # grasp_no_event.clear()
        print(f"============== episode {last_idx} done =========================")

        if state == "exit":
            break
except KeyboardInterrupt:
    exit_event.set()
finally:
    capture_keyboard.close()
    print("Exiting teleoperation recording.")
    if camera_preview is not None:
        camera_preview.close()
    if tactile_plotter is not None:
        tactile_plotter.close()
    if getattr(cs, "save_path", None) is not None:
        cs.stop()
    cs.end()
    if pedal_state is not None:
        pedal_state.close()
