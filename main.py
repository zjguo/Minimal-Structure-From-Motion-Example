from applications.sfm import *
from camera.camera_intrinsics import *

frames_seq = FrameSequence.from_frames_directory("test_data/SFM")

camera_intrinsics = CameraIntrinsics(
    focal_length=1901.6,
    principal_point=(1051.3, 778.61),
    horz_bias_factor=0.9965,
    sheer_factor=0)

sfm = SFM(frames_seq, camera_intrinsics)
sfm.run()