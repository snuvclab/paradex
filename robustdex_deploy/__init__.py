"""Real-robot deployment of the RobustDexGrasp teacher policy (xArm6 + Allegro v5) trained in MuJoCo-Warp.

Pure numpy + mujoco (kinematics only); no torch. The observation is rebuilt from the real joint states, the tracked
object point cloud (robot base frame, from paradex stereo_track) and a contact proxy, with exactly the definitions
used in training (see export_deploy.py / meta.json)."""
