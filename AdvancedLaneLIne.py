# Importing libraries
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt

# Using chess board images to obtain image points and object points to
# calibrate camera and remove distortion from the image

def points ():
  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Prepare object points
  objp = np.zeros((6*9,3), np.float32)
  objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

  # Arrays to store object points and image points
  # from the calibration images
  objpoints = [] # 3D points in real world space
  imgpoints = [] # 2D points in image plane

  # Make a list of calibration images
  images = glob.glob('./camera_cal_images/*.jpg')

  # fig, axs = plt.subplots(5,4, figsize=(16, 11))
  # fig.subplots_adjust(hspace = .2, wspace=.001)
  # axs = axs.ravel()

  # Go through the list and seach for chess board corners
  for i, fname in enumerate(images):
    img = cv2.imread(fname) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
      objpoints.append(objp)

      # Refining image points
      corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
      imgpoints.append(corners2)

      # Draw and display the corners
      img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
      # axs[i].axis('off')
      # axs[i].imshow(img)
      # cv2.imshow('img',img)
      # cv2.waitKey(100)

  cv2.destroyAllWindows()
  return objpoints, imgpoints, corners, ret

def calibration(objpoints, imgpoints):
  # Read an image camera_cal_images
  img = cv2.imread('./input_images/test6.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_size = (img.shape[1], img.shape[0])

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

  return mtx, dist, img
# './input_images/test6.jpg'
# './camera_cal_images/calibration1.jpg'
def undistortion(mtx, dist):
  img = cv2.imread('./input_images/test6.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  undist = cv2.undistort(img, mtx, dist, None, mtx)

  return undist

# objpoints, imgpoints, corners = points()

# # print(objpoints)
# # print(imgpoints)
# mtx, dist, img = calibration(objpoints, imgpoints)

# # print("mtx: ",mtx)
# # print("dist: ",dist)
# undist = undistortion(mtx, dist)

# Visualize undistortion
# fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# axs[0].imshow(img)
# axs[0].set_title('Original Image', fontsize=30)

# axs[1].imshow(undist)
# axs[1].set_title('Undistorted Image', fontsize=30)

# fig.tight_layout()
# # mpimg.imsave("test-after.jpg", color_select)
# # plt.imsave("output_images/test_before2.jpg", img)
# # plt.imsave("output_images/test_after2.jpg", undist)
# plt.show()

# nx = 9 # the number of inside corners in x
# ny = 6 # the number of inside corners in y

#===================================================
def unwarp(undist, src, ret):
  # Convert to grayscale
  gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
  # Find the chessboard corners
  # ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
  # # If corners found: 
  # # if(corners):
  if ret == True:
    # a) draw corners
    # img = cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    # b) define 4 source points 
    offset = 450
    img_size = (gray.shape[1], gray.shape[0])
    # src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    # #Note: you could pick any four of the detected corners 
    # # as long as those four corners define a rectangle
    # #One especially smart way to do this would be to use four well-chosen
    # # corners that were automatically detected during the undistortion steps
    # #We recommend using the automatic detection of corners in your code
    # # c) define 4 destination points 
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                                     [offset, img_size[1]], 
                                     [img_size[0]-offset, img_size[1]]])    
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    
  return warped, M

src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
objpoints, imgpoints, corners, ret = points()

# print(objpoints)
# print(imgpoints)
mtx, dist, img = calibration(objpoints, imgpoints)

# print("mtx: ",mtx)
# print("dist: ",dist)
undist = undistortion(mtx, dist)

unwarped, M = unwarp(undist, src, ret)

fig, axs = plt.subplots(1, 3, figsize=(20, 10))

axs[0].imshow(img)
axs[0].set_title('Original Image', fontsize=30)

axs[1].imshow(undist)
axs[1].set_title('Undistorted Image', fontsize=30)

axs[2].imshow(unwarped)
axs[2].set_title('Unwarped Image', fontsize=30)

fig.tight_layout()
# mpimg.imsave("test-after.jpg", color_select)
# plt.imsave("output_images/test_before2.jpg", img)
# plt.imsave("output_images/test_after2.jpg", undist)
plt.show()