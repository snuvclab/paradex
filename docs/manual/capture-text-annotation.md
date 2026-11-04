# Per-episode text annotation

HRI capture_robot.py prompts once after cs.stop() and the immediate integrity report, before the next episode. Type one line (Korean/Unicode supported) and Enter to save <episode>/text_annotation.json with episode, text and UTC created_at. Empty Enter skips. The last episode also prompts when capture was ended with q. EOF skips without blocking.

Capture keyboard events and annotations share one terminal reader so competing input() calls cannot steal the text. During annotation entry c/s/q are ordinary text; afterwards they resume as capture keys. GUI event refresh continues while waiting. Existing annotation files are retained; an additional annotation receives a unique suffix. Annotation writes use a fsynced temporary file and atomic rename. No video/state/tactile files are changed. Restart an already-running capture_robot.py to load this change.

Verification: three no-hardware annotation tests (input routing, Unicode/existing file preservation, blank/closed-input skip); together with tactile regression tests all six passed on mainpc robot Python. No robot/capture process launched or restarted.
