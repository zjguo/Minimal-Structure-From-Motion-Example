from video.frame_sequence import *
from feature_detection.detector import *
from photogrammetry.estimation import *
from photogrammetry.triangulation import *
from camera.camera import*
from camera.camera_intrinsics import*
from camera.camera_extrinsics import*
import numpy as np
from visualization.pyplot_visualizer import *
import copy

class SFM():
    
    def __init__(self, 
                 frame_seq : FrameSequence,
                 camera_intrinsic: CameraIntrinsics):
        self.frames = frame_seq.frames
        self.camera_intrinsic = camera_intrinsic

    def run(self):
        prev_frame = next(self.frames)
        curr_frame = next(self.frames)
        current_camera_extrinsic = CameraExtrinsics(np.zeros((3,1)), np.diag(np.ones(3,)))
        if(prev_frame is not None and curr_frame is not None):

            # get matched points
            kp1, kp2, matches = detect_and_match_pair(prev_frame, curr_frame, 2000)
            matched_points1 = get_matched_point_matrix(kp1, [i.queryIdx for i in matches])
            matched_points2 = get_matched_point_matrix(kp2, [i.trainIdx for i in matches])

            # estimate essential matrix and relative pose
            E, inliers = estimate_essential_matrix(matched_points1, matched_points2, self.camera_intrinsic)
            inlier_points1 = matched_points1[:, inliers]
            inlier_points2 = matched_points2[:, inliers]
            t, R = get_realizable_relpose_from_essential_matrix(inlier_points1, inlier_points2, self.camera_intrinsic, self.camera_intrinsic, E)

            # update camera parameters with relative pose
            prev_camera_extrinsic = copy.deepcopy(current_camera_extrinsic)
            camera1 = Camera(self.camera_intrinsic, prev_camera_extrinsic)
            current_camera_extrinsic.translation += t
            current_camera_extrinsic.rotation = R @ current_camera_extrinsic.rotation
            camera2 = Camera(self.camera_intrinsic, current_camera_extrinsic)

            # detect and match more points for coverage and triangulate to find 3D points
            kp1, kp2, matches = detect_and_match_pair(prev_frame, curr_frame, 100000)
            matched_points1 = get_matched_point_matrix(kp1, [i.queryIdx for i in matches])
            matched_points2 = get_matched_point_matrix(kp2, [i.trainIdx for i in matches])
            world_points, valid_world_point_indices, score  = triangulate_world_points(matched_points1, matched_points2, camera1, camera2, triangulate_3D_midpoint, 100)
            world_points = world_points[:,valid_world_point_indices]
            
            # draw point cloud
            vis = PyplotVisualizer(5)
            vis.draw_camera(prev_camera_extrinsic.translation, prev_camera_extrinsic.rotation.T)
            vis.draw_camera(current_camera_extrinsic.translation, current_camera_extrinsic.rotation.T, face_color='purple')
            colors = prev_frame[matched_points1[1,valid_world_point_indices].astype(int),matched_points1[0,valid_world_point_indices].astype(int),::-1]/255
            vis.ax.scatter(world_points[0,:], world_points[1,:], world_points[2,:], s=3, alpha=0.75, c=colors)
            vis.show()
