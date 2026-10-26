"""Run from a source checkout: python3 src/camera/live_viewer.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paradex.io.camera_system.live_viewer import main

if __name__ == '__main__':
    main()
