import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_and_match_pair(img1, img2, n_features = 200):

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=15, fastThreshold=1, scoreType=0)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return kp1 ,kp2, matches

def get_matched_point_matrix(kp, matches_indices):

    points = np.array([kp[i].pt for i in range(len(kp))])
    points = np.concatenate((points.T, np.ones((1,points.shape[0]))), axis=0)
    matched_points = np.array([points[:,i] for i in matches_indices]).T

    return matched_points

