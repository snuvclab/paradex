#!/usr/bin/env python3
"""Compatibility entry point for the renamed hand-neutral pose UI.

Use ``pedal_pose_ui.py`` for new commands. This module remains so existing
Allegro capture scripts and shell commands continue to work.
"""

if __package__:
    from .pedal_pose_ui import *  # noqa: F401,F403
    from .pedal_pose_ui import main
else:
    from pedal_pose_ui import *  # noqa: F401,F403
    from pedal_pose_ui import main


if __name__ == "__main__":
    main()
