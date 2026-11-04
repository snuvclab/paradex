"""Exercise capture CLI setup without starting robot or camera connections."""

import atexit
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "src/dataset_acquisition/hri/capture_robot.py"


def test_direct_execution_can_find_local_package_without_installation(tmp_path):
    # Disable site packages to exclude editable installs and stop before hardware imports.
    probe = """
import builtins
import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType

script = Path(sys.argv[1])
sys.path.insert(0, str(script.parent))
sys.modules['numpy'] = ModuleType('numpy')
original_import = builtins.__import__

def check_import(name, *args, **kwargs):
    if name == 'paradex.dataset_acqusition.capture':
        spec = importlib.util.find_spec('paradex')
        assert spec is not None, 'Local paradex package not found'
        expected = script.parents[3] / 'paradex' / '__init__.py'
        assert Path(spec.origin).resolve() == expected.resolve()
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)

builtins.__import__ = check_import
runpy.run_path(str(script), run_name='__main__')
raise AssertionError('Package import was not reached')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _Configured(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize("side", ["right", "bimanual"])
def test_hand_none_disables_manus_gesture_control(monkeypatch, side):
    config = _configure(monkeypatch, "--hand", "None", "--hand-side", side)
    assert config["require_left_control"] is False
    assert config.get("hand") is None


@pytest.mark.parametrize("options,expected", [((), "split"), (("--teleop-receiver-mode", "shared"), "shared")])
def test_teleop_receiver_mode_is_forwarded(monkeypatch, options, expected):
    assert _configure(monkeypatch, *options)["teleop_receiver_mode"] == expected


def _configure(monkeypatch, *options, network_info=None):
    def capture_session(**kwargs):
        raise _Configured(kwargs)

    stub_modules = {
        "paradex.dataset_acqusition.capture": {"CaptureSession": capture_session},
        "paradex.utils.path": {"shared_dir": "/unused"},
        "paradex.utils.file_io": {"find_latest_index": lambda *_args: 0},
        "paradex.dataset_acqusition.text_annotation": {
            "CaptureKeyboard": lambda *_args: SimpleNamespace(start=lambda: None, close=lambda: None),
        },
        "paradex.utils.system": {
            "get_pc_list": lambda: [],
            "network_info": {} if network_info is None else network_info,
        },
        "paradex.io.robot_controller.inspire_f1_tactile_plotter": {
            "InspireF1RealtimeTactilePlotter": object,
        },
        "paradex.io.streamdeck_pedal": {
            "MiddlePedalState": lambda: SimpleNamespace(close=lambda: None),
        },
    }
    for name, attributes in stub_modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(atexit, "register", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--name", "test", *options])
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except _Configured as configured:
        return configured.kwargs
    pytest.fail("CaptureSession was not configured")


def test_unimanual_default_is_preserved(monkeypatch):
    config = _configure(monkeypatch)
    assert config["hand"] == "allegro_v5"
    assert "hand_left" not in config and "hand_right" not in config
    assert config["tactile"] is True
    assert config["hand_command_rate_hz"] == 30.0
    assert config["ip"] is False


def test_startup_profiling_is_opt_in(monkeypatch):
    config = _configure(monkeypatch)
    assert config["profile_startup"] is False
    config = _configure(monkeypatch, "--profile-startup")
    assert config["profile_startup"] is True


@pytest.mark.parametrize("options", [
    (),
    ("--hand-side", "bimanual", "--hand-left", "inspire_dftp", "--hand-right", "allegro_v5"),
])
def test_zero_allegro_command_rate_disables_capture_target_limit(monkeypatch, options):
    config = _configure(monkeypatch, *options, "--allegro-command-rate-hz", "0")
    assert config["hand_command_rate_hz"] is None


@pytest.mark.parametrize("rate", ["-1", "nan", "inf", "-inf"])
def test_invalid_allegro_command_rate_is_rejected(monkeypatch, rate):
    with pytest.raises(SystemExit) as exc:
        _configure(monkeypatch, f"--allegro-command-rate-hz={rate}")
    assert exc.value.code == 2


def test_bimanual_common_hand_default_is_preserved(monkeypatch):
    config = _configure(monkeypatch, "--hand-side", "bimanual", "--hand", "inspire")
    assert config["hand"] is None
    assert config["hand_left"] == config["hand_right"] == "inspire"
    assert set(config["hand_kwargs"]) == {"left", "right"}
    assert config["ip"] is True
    assert config["tactile"] is False
    assert config["hand_kwargs"]["left"]["host"] == "192.168.11.210"
    assert config["hand_kwargs"]["left"]["interface"] == "enp8s0f3"
    assert config["hand_kwargs"]["right"]["host"] == "192.168.12.210"
    assert config["hand_kwargs"]["right"]["interface"] == "enp8s0f1"


@pytest.mark.parametrize("hand", ["inspire", "inspire_dftp"])
def test_left_inspire_on_right_port_keeps_logical_side_and_force_limits(monkeypatch, hand):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual", "--hand-left", hand,
        network_info={"inspire_ip": {"left": {
            "ip": "192.168.11.210", "interface": "enp8s0f1",
        }}},
    )

    assert config["hand_left"] == hand
    assert config["hand_right"] == "allegro_v5"
    assert config["hand_kwargs"] == {"left": {
        "host": "192.168.11.210", "interface": "enp8s0f1",
        "force_limits": [400, 400, 400, 400, 400, 400],
    }}
    assert config["ip"] is True


def test_explicit_inspire_endpoint_overrides_machine_wiring(monkeypatch):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual", "--hand", "inspire",
        "--inspire-left-interface", "left-test", "--inspire-left-ip", "192.0.2.10",
        "--inspire-right-interface", "right-test", "--inspire-right-ip", "192.0.2.20",
        network_info={"inspire_ip": {
            "left": {"ip": "192.168.11.210", "interface": "enp8s0f1"},
            "right": {"ip": "192.168.12.210", "interface": "enp8s0f3"},
        }},
    )

    assert config["hand_kwargs"]["left"]["host"] == "192.0.2.10"
    assert config["hand_kwargs"]["left"]["interface"] == "left-test"
    assert config["hand_kwargs"]["right"]["host"] == "192.0.2.20"
    assert config["hand_kwargs"]["right"]["interface"] == "right-test"


def test_swapped_inspire_ports_preserve_both_ips_and_side_specific_force(monkeypatch):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual", "--hand", "inspire",
        network_info={"inspire_ip": {
            "left": {"interface": "enp8s0f1"},
            "right": {"interface": "enp8s0f3"},
        }},
    )

    assert config["hand_kwargs"] == {
        "left": {"host": "192.168.11.210", "interface": "enp8s0f1",
                 "force_limits": [400, 400, 400, 400, 400, 400]},
        "right": {"host": "192.168.12.210", "interface": "enp8s0f3",
                  "force_limits": [400, 400, 400, 400, 600, 400]},
    }


@pytest.mark.parametrize("options", [("--no-vive",), ("--no-vive", "--pedal")])
def test_left_only_inspire_uses_same_machine_wiring_as_bimanual(monkeypatch, options):
    config = _configure(
        monkeypatch, "--hand-side", "left", "--hand", "inspire", *options,
        network_info={"inspire_ip": {"left": {"interface": "enp8s0f1"}}},
    )

    assert config["hand_side"] == "left"
    assert config["hand"] == "inspire"
    assert config["hand_kwargs"] == {
        "host": "192.168.11.210", "interface": "enp8s0f1",
        "force_limits": [400, 400, 400, 400, 400, 400],
    }
    assert config["ip"] is True
    assert config["use_vive"] == ("--no-vive" not in options)
    assert config["require_left_control"] is False


@pytest.mark.parametrize("options", [(), ("--ip",)])
def test_right_unimanual_inspire_preserves_existing_transport_defaults(monkeypatch, options):
    config = _configure(monkeypatch, "--hand", "inspire", *options)

    assert config["hand_kwargs"] is None
    assert config["ip"] == ("--ip" in options)


@pytest.mark.parametrize("side", ["left", "right"])
def test_mixed_hands_inherit_unspecified_side_and_configure_only_inspire(monkeypatch, side):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual", f"--hand-{side}", "inspire",
        f"--inspire-{side}-ip", "192.0.2.10",
        f"--inspire-{side}-interface", "test0",
        f"--inspire-{side}-force-limits", "100,200,300,400,500,600",
        "--allegro-command-rate-hz", "25",
    )
    other_side = "right" if side == "left" else "left"
    assert config["hand"] is None
    assert config[f"hand_{side}"] == "inspire"
    assert config[f"hand_{other_side}"] == "allegro_v5"
    assert config["hand_kwargs"] == {
        side: {"host": "192.0.2.10", "interface": "test0", "force_limits": [100, 200, 300, 400, 500, 600]},
    }
    assert config["ip"] is True
    assert config["tactile"] is True
    assert config["hand_command_rate_hz"] == 25.0


def test_both_overrides_replace_common_default_and_accept_underscore_aliases(monkeypatch):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual",
        "--hand_left", "wuji_direct", "--hand_right", "inspire_f1",
    )
    assert config["hand_left"] == "wuji_direct"
    assert config["hand_right"] == "inspire_f1"
    assert config["tactile"] is False
    assert config["hand_command_rate_hz"] is None
    assert config["hand_kwargs"] is None
    assert config["ip"] is False


@pytest.mark.parametrize("disabled", ["none", "null", ""])
def test_explicit_none_disables_side_without_falling_back(monkeypatch, disabled):
    config = _configure(monkeypatch, "--hand-side", "bimanual", "--hand-left", disabled)
    assert config["hand"] is None
    assert config["hand_left"] is None
    assert config["hand_right"] == "allegro_v5"


@pytest.mark.parametrize("option", ["--hand-left", "--hand-right"])
def test_side_overrides_require_bimanual(monkeypatch, capsys, option):
    with pytest.raises(SystemExit) as exc:
        _configure(monkeypatch, option, "inspire")
    assert exc.value.code == 2
    assert "require --hand-side bimanual" in capsys.readouterr().err


def test_right_allegro_dexpilot_allows_mixed_bimanual(monkeypatch):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual",
        "--hand-left", "inspire", "--allegro-retarget", "dexpilot",
    )
    assert config["hand_right"] == "allegro_v5"
    assert config["allegro_retarget"] == "dexpilot"


def test_left_allegro_dexpilot_is_rejected(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _configure(
            monkeypatch, "--hand-side", "bimanual",
            "--hand-right", "inspire", "--allegro-retarget", "dexpilot",
        )
    assert exc.value.code == 2
    assert "no left Allegro V5 hand" in capsys.readouterr().err


def test_no_tactile_overrides_allegro_default_in_mixed_configuration(monkeypatch):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual", "--hand-left", "inspire", "--no-tactile",
    )
    assert config["tactile"] is False


@pytest.mark.parametrize("options,expected", [((), "split"), (("--teleop-control-mode", "shared"), "shared")])
def test_teleop_control_mode_is_forwarded(monkeypatch, options, expected):
    assert _configure(monkeypatch, *options)["teleop_control_mode"] == expected


def test_august_18_allegro_mode_is_forwarded_in_mixed_bimanual(monkeypatch):
    config = _configure(
        monkeypatch, "--hand-side", "bimanual", "--hand-left", "inspire_dftp",
        "--hand-right", "allegro_v5", "--allegro-retarget", "legacy-0818",
    )
    assert config["allegro_retarget"] == "legacy-0818"
    assert config["hand_left"] == "inspire_dftp"
    assert config["hand_right"] == "allegro_v5"


@pytest.mark.parametrize("options,expected", [((), "linear"), (("--allegro-thumb-distal-mapping", "sine"), "sine")])
def test_allegro_thumb_distal_mapping_is_forwarded(monkeypatch, options, expected):
    assert _configure(monkeypatch, *options)["allegro_thumb_distal_mapping"] == expected
