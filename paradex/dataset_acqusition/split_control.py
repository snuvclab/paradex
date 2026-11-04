"""Independent arm/hand target loops; drivers retain their own publish loops."""
from threading import Event, Thread
import time

import chime
import numpy as np


def run_split_teleop(session, events, state_policy, loop_callback,
                    bimanual_state_provider, unimanual_state_provider):
    # Import here: capture exposes the shared safety policy without a circular
    # import during module initialization.
    from paradex.dataset_acqusition.capture import (
        _PoseCommandLimiter, _allegro_v5_feedback_hold_target,
        _is_allegro_ui_aligned_hand,
    )

    previous = getattr(session, "_control_threads", ())
    if any(thread.is_alive() for thread in previous):
        raise RuntimeError("Previous teleop control loop has not stopped")
    sides = ("Left", "Right") if session.hand_side == "Bimanual" else (session.hand_side,)
    arms = {side: getattr(session, "arm_" + side.lower())
            if session.hand_side == "Bimanual" else session.arm for side in sides}
    hands = {side: getattr(session, "hand_" + side.lower())
             if session.hand_side == "Bimanual" else session.hand for side in sides}
    names = {side: getattr(session, "hand_name_" + side.lower(), None)
             if session.hand_side == "Bimanual" else getattr(session, "hand_name", None)
             for side in sides}
    home = {side: arm.get_data()["position"] if arm is not None else np.eye(4)
            for side, arm in arms.items()}
    session.retargetor.start(home if len(sides) == 2 else home[sides[0]])
    stop = Event()
    # Tuple replacement is atomic under CPython. No controller/retargeter lock
    # is shared between these loops. The generation invalidates work computed
    # before a pedal transition, including a release followed by a re-press.
    control = (0, 1, False)
    failures = {}

    def permitted(token, arm=False):
        return not stop.is_set() and control == token and token[1] == 0 and (not arm or token[2])

    def arm_loop():
        was_enabled = False
        previous_generation = None
        limiters = {}
        active = [side for side in sides if arms[side] is not None]
        while not stop.is_set():
            token = control
            if not permitted(token, arm=True):
                was_enabled = False
                stop.wait(0.01)
                continue
            data = session.teleop_device.get_arm_data()
            if not all(data.get(side) is not None for side in active):
                stop.wait(0.01)
                continue
            if not was_enabled or previous_generation != token[0]:
                current = {side: arms[side].get_data()["position"] if arms[side] is not None else np.eye(4)
                           for side in sides}
                session.retargetor.start_arm(current if len(sides) == 2 else current[sides[0]])
                limiters = {side: _PoseCommandLimiter(
                    current[side], time.monotonic(),
                    smoothing_tau_s=getattr(session, "vive_arm_smoothing_tau_s", 0.08),
                ) for side in active}
                was_enabled = True
                previous_generation = token[0]
            # In bimanual sessions the mapper expects both wrist keys, including
            # an absent controller's wrist. Hold if either tracker is missing.
            if not all(data.get(side) is not None for side in sides):
                stop.wait(0.01)
                continue
            actions = session.retargetor.get_arm_action(data)
            actions = actions if len(sides) == 2 else (actions,)
            for side, action in zip(sides, actions):
                if side not in active or not permitted(token, arm=True):
                    continue
                limiter = limiters[side]
                pose, translation, rotation = limiter.filter(action, time.monotonic())
                if pose is not None:
                    if permitted(token, arm=True):
                        arms[side].move(pose)
                elif limiter.rejected_count == 1 or limiter.rejected_count % 100 == 0:
                    print(f"Rejected {side} VIVE arm command: translation={translation * 1000:.1f} mm, rotation={rotation:.1f} deg")
            stop.wait(0.01)

    def hand_loop():
        held = {}
        previous_token = None
        provider = getattr(session, "hand_action_provider", None)
        while not stop.is_set():
            token = control
            if token != previous_token:
                held = {}
                previous_token = token
            if token[1] == 0:
                held = {}
                data = session.teleop_device.get_hand_data()
                required = [side for side in sides if hands[side] is not None]
                if provider is None and not all(data.get(side) is not None for side in required):
                    stop.wait(0.01)
                    continue
                # Unconfigured sides have no retargeter and need no MANUS frame.
                actions = provider() if provider is not None else session.retargetor.get_hand_action(data)
                actions = actions if len(sides) == 2 else (actions,)
                session._move_hands_if_due(
                    tuple((hands[side], action, names[side]) for side, action in zip(sides, actions)),
                    teleop_data=data, state=0, command_allowed=lambda: permitted(token),
                )
            elif token[1] == 1:
                # Preserve the Allegro feedback hold on pedal release. This
                # feedback read is confined to the hand loop as well.
                for side in sides:
                    if hands[side] is not None and _is_allegro_ui_aligned_hand(names[side]):
                        if side not in held:
                            target = _allegro_v5_feedback_hold_target(hands[side].get_data())
                            if target is not None:
                                held[side] = target
                session._move_hands_if_due(
                    tuple((hands[side], target, names[side]) for side, target in held.items()),
                    state=1, command_allowed=lambda: not stop.is_set() and control == token,
                )
            stop.wait(0.01)

    def guarded(name, target):
        try:
            target()
        except Exception as exc:
            failures[name] = exc
            print(f"{name} teleop loop stopped; other loop continues: {exc}", flush=True)

    workers = []
    if any(arm is not None for arm in arms.values()) and getattr(session, "use_vive", True):
        workers.append(Thread(target=guarded, args=("Arm", arm_loop), name="teleop-arm-targets", daemon=True))
    if any(hand is not None for hand in hands.values()):
        workers.append(Thread(target=guarded, args=("Hand", hand_loop), name="teleop-hand-targets", daemon=True))
    session._control_threads = workers
    session._control_failures = failures
    chime.warning(sync=True)
    print("Teleop command loops: split arm/hand threads; independent VIVE/MANUS freshness", flush=True)
    stop_count = exit_count = 0
    previous_state = None
    try:
        for worker in workers:
            worker.start()
        while True:
            if events is not None:
                if events["exit"].is_set():
                    return "exit"
                if session.save_path is None and events["save"].is_set():
                    return "start"
                if session.save_path is not None and events["stop"].is_set():
                    return "stop"
            if loop_callback is not None:
                try:
                    loop_callback(session)
                except Exception as exc:
                    print(f"teleop loop_callback failed: {exc}")
                    loop_callback = None
            provider = bimanual_state_provider or unimanual_state_provider
            if provider is not None:
                try:
                    state = provider()
                except Exception as exc:
                    print(f"Pedal state provider failed; pausing teleop: {exc}")
                    state = 1
                if state not in (0, 1):
                    raise ValueError(f"Teleop state provider returned invalid state: {state}")
            elif len(sides) == 2:
                state = 0
            else:
                state = session.teleop_device.get_state()
                if state is None:
                    state = 0 if state_policy == "keyboard_control" else 1
            arm_enabled = True
            arm_provider = getattr(session, "arm_command_enabled_provider", None)
            if arm_provider is not None:
                try:
                    arm_enabled = bool(arm_provider())
                except Exception as exc:
                    print(f"Arm command enable provider failed; holding arm: {exc}")
                    arm_enabled = False
            value = (state, arm_enabled)
            if value != previous_state:
                control = (control[0] + 1, state, arm_enabled)
                previous_state = value
            if session.save_path is not None:
                session.state_hist.append(state)
                session.state_time.append(time.time())
                # Retain the synchronized recording format and its original
                # freshness rule. It no longer gates either command loop.
                session.teleop_device.get_data()
            if state_policy == "gesture_control" and len(sides) == 1:
                stop_count = stop_count + 1 if state == 2 else 0
                exit_count = exit_count + 1 if state == 3 else 0
                if exit_count > 90:
                    return "exit"
                if stop_count > 90:
                    return "stop"
            if workers and len(failures) == len(workers):
                raise RuntimeError("All teleop control loops stopped")
            stop.wait(0.01)
    finally:
        stop.set()
        control = (control[0] + 1, 1, False)
        for worker in workers:
            worker.join(timeout=2.0)
        if any(worker.is_alive() for worker in workers):
            raise RuntimeError("Teleop control loop did not stop; refusing to restart it")
