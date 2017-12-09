#=============================================================================
#=== Importing libraries =====================================================
#=============================================================================

import numpy as np
import cv2
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import imageio
imageio.plugins.ffmpeg.download()
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#=============================================================================
#=== Camera Calibration ======================================================
#=============================================================================

# Using chess board images to obtain image points and object points to
# calibrate camera and remove distortion from the image

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points
# from the calibration images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Read and make a list of calibration images
images = glob.glob('./camera_cal_images/*.jpg')

# Go through the list and search for chess board corners
for i, fname in enumerate(images):

  # Read in each image
  img = cv2.imread(fname) 

  fig, axs = plt.subplots(5,4, figsize=(16,11))
  fig.subplots_adjust(hspace = .2, wspace = 0.001)
  axs.ravel()
  # Convert image to grayscale
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Find the chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

  # If corners are found, add object points, image points
  if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    # Draw corners on the chessboard
    drawimg = cv2.drawChessboardCorners(img, (9,6), corners, ret)
 
    # Show the chessboard images with the corners drawn on them
  # axs[i].axis('off')
  # axs[i].imshow(drawimg)
    # cv2.imshow('FRAME',drawimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Camera calibration, it takes in object, image points and shape of the input image ..
# and returns the distortion coefs (dist), camera matrix to transform 3D obj points to 2D image points ..
# also returns the position of the camera in the world (rotation and translational matrices (rvecs, tvecs))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('Camera calibration done!')

for fname in images:
  img = cv2.imread(fname) 
  # Undistorting the images
  dst = cv2.undistort(img, mtx, dist, None, mtx)

  cv2.imshow('FRAME2',dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

print('Undistortion done!')
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# dist_pickle = {}
# dist_pickle["mtx"] = mtx
# dist_pickle["dist"] = dist
# pickle.dump( dist_pickle, open( "calibration.p", "wb" ) )
# './input_images/test6.jpg'
# './camera_cal_images/calibration1.jpg'