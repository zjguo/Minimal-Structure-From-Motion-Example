# Minimal Structure From Motion Example

Two view example is provided. Opencv is used for feature detection and matching as well as essential matrix estimation. 
However, there is code provided if you want to estimate essential matrix without opencv.
Useful as a starting point for more advanced projects.

Code is provided for:
- fundamental matrix estimation (RANSAC)
- essential matrix from fundamental matrix (with normalization)
- extract realizable relative pose from essential matrix
- mid-point triangulation
- camera matrices
- visualization

![SFM_Example_views](https://user-images.githubusercontent.com/46606255/214897260-42b68d5e-c419-479c-b866-006c78589045.PNG)
*Two Undistorted Views obtained from: https://www.mathworks.com/help/vision/ug/structure-from-motion-from-two-views.html*

![SFM_Example](https://user-images.githubusercontent.com/46606255/214894796-0a80a275-3e02-4100-95b7-967c3f74d45d.PNG)

*Point cloud obtained from above views with camera positions*

# Requirements
**Not hard version requirements, just what I used.**
- python 3.9
- opencv 4.7
- numpy
- pyplot for visualization

# Usage
python main.py

*Might take a while. If taking too long, reduce number of detected keypoints in applications/sfm.py


