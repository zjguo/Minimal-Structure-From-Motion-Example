import numpy as np
from nptyping import NDArray, Shape, Float
from .camera_intrinsics import CameraIntrinsics
from .camera_extrinsics import CameraExtrinsics

class Camera():
    
    def __init__(self, 
                 camera_intrinsics : CameraIntrinsics,
                 camera_extrinsincs : CameraExtrinsics):
                 
        self.K = camera_intrinsics.K
        self.translation = camera_extrinsincs.translation
        self.rotation_matrix = camera_extrinsincs.rotation
        
    def direct_linear_transform(self, 
                                homo_point_coords : NDArray[Shape["4,1"], Float]) -> NDArray[Shape["4,1"], Float]:

        translation_matrix = np.array([[1, 0, 0, -self.translation[0]],
                                       [0, 1, 0, -self.translation[1]],
                                       [0, 0, 1, -self.translation[2]]])

        return self.K @ self.rotation_matrix @ translation_matrix @ homo_point_coords
        