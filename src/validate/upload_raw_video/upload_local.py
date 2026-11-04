"""Local upload entry point using the same verified upload and automatic cleanup pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paradex.video.safe_raw_video_processor import main

if __name__ == '__main__':
    main()
