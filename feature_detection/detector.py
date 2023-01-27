import cv2 as cv
import numpy as np

def detect_and_match_pair(
    img1, 
    img2, 
    n_features = 200,
    scale_factor = 1.2,
    n_levels=15,
    fast_threshold=31,
    score_type=0,
    detector_type='orb',
    lowes_ratio = 0.75):

    # get keypoints
    kpts1 = []
    kpts2 = []
    orb = cv.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels, fastThreshold=fast_threshold, scoreType=score_type)
    if detector_type == 'orb':
        kpts1 = orb.detect(img1, None)
        kpts2 = orb.detect(img2, None)
    elif detector_type == 'gftt':
        locs1 = cv.goodFeaturesToTrack(np.mean(img1, axis=2).astype(np.uint8), n_features, qualityLevel=0.01, minDistance=7)
        kpts1 = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in locs1]
        locs2 = cv.goodFeaturesToTrack(np.mean(img2, axis=2).astype(np.uint8), n_features, qualityLevel=0.01, minDistance=7)
        kpts2 = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in locs2]
    
    # find the descriptors with ORB
    kpts1, des1 = orb.compute(img1,kpts1)
    kpts2, des2 = orb.compute(img2,kpts2)

    # lowes ratio test
    good_matches = []
    if lowes_ratio >= 1.0:
        # Match descriptors.
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = matches
    else:
        # Match descriptors.
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        for m,n in matches:
            if m.distance < lowes_ratio*n.distance:
                good_matches.append(m)

    good_matches = sorted(good_matches, key = lambda x:x.distance)
    return kpts1 ,kpts2, good_matches

def get_matched_point_matrix(kp, matches_indices):

    points = np.array([kp[i].pt for i in range(len(kp))])
    points = np.concatenate((points.T, np.ones((1,points.shape[0]))), axis=0)
    matched_points = np.array([points[:,i] for i in matches_indices]).T

    return matched_points

