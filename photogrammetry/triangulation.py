from nptyping import NDArray, Float, Shape
import numpy as np
from typing import Tuple
from camera.camera import *
from camera.camera_intrinsics import *
from camera.camera_extrinsics import *
from typing import Callable

def triangulate_world_points(points1 : NDArray[Shape["3,Any"], Float], 
                             points2 : NDArray[Shape["3,Any"], Float], 
                             camera1 : Camera, 
                             camera2 : Camera,
                             triangulation_method : Callable,
                             reprojection_error_threshold : Float = 0.5
                             ) -> Tuple[NDArray[Shape["3,Any"], Float], list[int], Float]:
    
    if points1.shape[1] != points2.shape[1]:
        raise Exception("Must have the same number of points!")

    world_points = np.zeros((3,points1.shape[1]))
    for i in range(points1.shape[1]):
        world_point = triangulation_method(camera1.translation, 
                                           camera2.translation,
                                           camera1.rotation_matrix.T @ np.linalg.inv(camera1.K) @ np.expand_dims(points1[:,i], axis=1),
                                           camera2.rotation_matrix.T @ np.linalg.inv(camera2.K) @ np.expand_dims(points2[:,i], axis=1))

        world_points[:,i] = [x for x in world_point]

    # calculate reprojection errors
    homo_world_points = np.append(world_points, np.ones((1,world_points.shape[1])), axis=0)
    reprojected_points1 = camera1.direct_linear_transform(homo_world_points)
    reprojected_points1 = reprojected_points1 / reprojected_points1[2,:]
    reprojected_points2 = camera2.direct_linear_transform(homo_world_points)
    reprojected_points2 = reprojected_points2 / reprojected_points2[2,:]
    points1_errors = np.sum(np.power(reprojected_points1 - points1,2), axis=0)
    points2_errors = np.sum(np.power(reprojected_points2 - points2,2), axis=0)
    valid_world_point_indices = (points1_errors < reprojection_error_threshold) & (points2_errors < reprojection_error_threshold)
    avg_valid_reprojection_error = []
    if len(valid_world_point_indices) == 0:
        avg_valid_reprojection_error = -1
    else:
        avg_valid_reprojection_error = np.sum((points1_errors[valid_world_point_indices] + points2_errors[valid_world_point_indices]))/len(valid_world_point_indices)
    
    return world_points, valid_world_point_indices, avg_valid_reprojection_error

def triangulate_3D_midpoint(
    starting_point1: NDArray[Shape["3,1"], Float],
    starting_point2: NDArray[Shape["3,1"], Float],
    direction1 : NDArray[Shape["3,1"], Float], 
    direction2 : NDArray[Shape["3,1"], Float],
    ) -> NDArray[Shape["3,1"], Float]:
        
    # form system of equations matrix Ax=b
    A = np.array([[(direction1.T @ direction1).item(), (-direction2.T @ direction1).item()],
                   [-(direction1.T @ direction2).item(), (direction2.T @ direction2).item()]])

    b = np.array([((starting_point2 - starting_point1).T @ direction1).item(), ((starting_point1 - starting_point2).T @ direction2).item()])
    x = np.dot(np.linalg.inv(A), b)

    # get optimal point in each direction
    op_point1 = starting_point1 + x[0]*direction1
    op_point2 = starting_point2 + x[1]*direction2

    # find mid-point
    mid_point = (op_point1 + op_point2) / 2
    
    return mid_point

def triangulate_3D_linear_least_squares(
    starting_point1: NDArray[Shape["3,1"], Float],
    starting_point2: NDArray[Shape["3,1"], Float],
    direction1 : NDArray[Shape["3,1"], Float], 
    direction2 : NDArray[Shape["3,1"], Float],
    ) -> NDArray[Shape["3,1"], Float]:

    raise NotImplementedError

def triangulate_3D_nonlinear_least_squares(
    starting_point1: NDArray[Shape["3,1"], Float],
    starting_point2: NDArray[Shape["3,1"], Float],
    direction1 : NDArray[Shape["3,1"], Float], 
    direction2 : NDArray[Shape["3,1"], Float],
    ) -> NDArray[Shape["3,1"], Float]:

    raise NotImplementedError

def triangulate_3D_optimal(
    starting_point1: NDArray[Shape["3,1"], Float],
    starting_point2: NDArray[Shape["3,1"], Float],
    direction1 : NDArray[Shape["3,1"], Float], 
    direction2 : NDArray[Shape["3,1"], Float],
    ) -> NDArray[Shape["3,1"], Float]:

    raise NotImplementedError