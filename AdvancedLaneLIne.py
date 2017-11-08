# Importing libraries
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt


#=== Camera Calibration ======================================================

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
  img = cv2.imread('./input_images/test2.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_size = (img.shape[1], img.shape[0])

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

  return mtx, dist, img
# './input_images/test6.jpg'
# './camera_cal_images/calibration1.jpg'

#=== Distortion Correction ======================================================
def undistortion(mtx, dist):
  img = cv2.imread('./input_images/test2.jpg')
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

#=== Perspective Transform ================================================
def unwarp(undist, src, dst, ret):
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
    # offset = 450
    img_size = (gray.shape[1], gray.shape[0])
    # src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    # #Note: you could pick any four of the detected corners 
    # # as long as those four corners define a rectangle
    # #One especially smart way to do this would be to use four well-chosen
    # # corners that were automatically detected during the undistortion steps
    # #We recommend using the automatic detection of corners in your code
    # # c) define 4 destination points 

    

    # print(src)
    # print(dst)
	# dst = np.float32([[(img_size[0]/4), 0],
	# 	[(img_size[0]/4), img_size[1]],
	# 	[(img_size[0]*3/4), img_size[1]],
	# 	[(img_size[0]*3/4), 0]])

    # dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
    #                                  [offset, img_size[1]], 
    #                                  [img_size[0]-offset, img_size[1]]])    
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    
  return warped, M

#=======================================================================
#=== Gradient Threshold ================================================
#=======================================================================

def abs_sobel_thresh(img, orient, thresh_min=25, thresh_max=255):
    # Calculate directional gradient
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    # 3) Calculate the magnitude 
    sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaledSobel = np.uint8(255*sobelxy/np.max(sobelxy))
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(scaledSobel)
    mag_binary[(scaledSobel >= mag_thresh[0]) & (scaledSobel <= mag_thresh[1])] = 1
    # Return the binary image
    return mag_binary

def dir_threshold(img, sobel_kernel=7, thresh=(0, 0.09)):
    # Calculate gradient direction
    # Apply threshold
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    gradDir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gradDir)
    dir_binary[(gradDir >= thresh[0]) & (gradDir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    # Return the binary image
    return dir_binary

#=======================================================================
#=== Color Threshold ================================================
#=======================================================================

# Edit this function to create your own pipeline.
def hls_threshold(img, h_thresh=(15, 100), l_thresh=(10, 150), s_thresh=(90, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,1]
    H = H*(255/np.max(H))
    L = hls[:,:,0]
    L = L*(255/np.max(L))
    S = hls[:,:,2]
    S = S*(255/np.max(S))
    # 2) Apply a threshold to the channels
    h_binary = np.zeros_like(H)
    h_binary[(H > h_thresh[0]) & (H <= h_thresh[1])] = 1

    l_binary = np.zeros_like(L)
    l_binary[(L > l_thresh[0]) & (L <= l_thresh[1])] = 1

    s_binary = np.zeros_like(S)
    s_binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1

    return h_binary, l_binary, s_binary
    
#===================================================================
#=== Function Calls ================================================
#===================================================================

# src = np.float32([(575,464),
#                   (707,464), 
#                   (258,682), 
#                   (1049,682)])
objpoints, imgpoints, corners, ret = points()

# print(objpoints)
# print(imgpoints)
# print(corners)
mtx, dist, img = calibration(objpoints, imgpoints)

# print("mtx: ",mtx)
# print("dist: ",dist)
undist = undistortion(mtx, dist)

img_size = (img.shape[1], img.shape[0])

# src = np.float32([((img_size[0]/2)-55,(img_size[1]/2)+100),
#                   ((img_size[0]/6)-10,img_size[1]), 
#                   ((img_size[0]*5/6)+60,img_size[1]), 
#                   ((img_size[0]/2)+55,(img_size[1]/2)+100)])
print(img_size)
src = np.float32([((img_size[0]/2)-65,(img_size[1]/2)+100),
                  ((img_size[0]/6)-10,img_size[1]), 
                  ((img_size[0]*5/6)+100,img_size[1]), 
                  ((img_size[0]/2)+55,(img_size[1]/2)+100)])

dst = np.float32([((img_size[0]/4),0),
                  (img_size[0]/4,img_size[1]), 
                  ((img_size[0]*3)/4,img_size[1]), 
                  ((img_size[0]*3)/4,0)])

unwarped, M = unwarp(undist, src, dst, ret)

# Choose a Sobel kernel size
# ksize = 3 # Choose a larger odd number to smooth gradient measurements


# Apply each of the thresholding functions
grad_binary_x = abs_sobel_thresh(unwarped, orient='x')
grad_binary_y = abs_sobel_thresh(unwarped, orient='y')
mag_binary = mag_thresh(unwarped)
dir_binary = dir_threshold(unwarped)

combined = np.zeros_like(mag_binary)
# combined[((mag_binary == 1) & (dir_binary == 1))] = 1
combined[((grad_binary_x == 1) & (grad_binary_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
# Visualize sobel absolute threshold
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.2)
ax1.imshow(unwarped)
ax1.set_title('Unwarped Image', fontsize=10)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Gradient Magnitude', fontsize=10)
ax3.imshow(dir_binary, cmap='gray')
ax3.set_title('Gradient Direction', fontsize=10)
ax4.imshow(combined, cmap='gray')
ax4.set_title('Combined Gradient (mag + dir)', fontsize=10)


h_binary, l_binary, s_binary = hls_threshold(unwarped)

g, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
g.subplots_adjust(hspace = .2, wspace=.2)
ax1.imshow(unwarped)
ax1.set_title('Unwarped Image', fontsize=10)
ax2.imshow(h_binary, cmap='gray')
ax2.set_title('H-Channel', fontsize=10)
ax3.imshow(l_binary, cmap='gray')
ax3.set_title('L-Channel', fontsize=10)
ax4.imshow(s_binary, cmap='gray')
ax4.set_title('S-Channel', fontsize=10)

# fig, axs = plt.subplots(1, 3, figsize=(20, 10))

# axs[0].imshow(img)
# axs[0].set_title('Original Image', fontsize=30)

# axs[1].imshow(undist)
# x1 = [src[0][0],src[1][0],src[2][0],src[3][0],src[0][0]]
# y1 = [src[0][1],src[1][1],src[2][1],src[3][1],src[0][1]]
# axs[1].plot(x1,y1,color='red',linewidth=3)
# axs[1].set_title('Undistorted Image', fontsize=30)

# axs[2].imshow(unwarped)
# x2 = [dst[0][0],dst[1][0],dst[2][0],dst[3][0],dst[0][0]]
# y2 = [dst[0][1],dst[1][1],dst[2][1],dst[3][1],dst[0][1]]
# axs[2].plot(x2,y2,color='red',linewidth=3)
# axs[2].set_title('Unwarped Image', fontsize=30)

# fig.tight_layout()
# # mpimg.imsave("test-after.jpg", color_select)
# # plt.imsave("output_images/test_before2.jpg", img)
# # plt.imsave("output_images/test_after2.jpg", undist)
# plt.show()


# Visualize multiple color space channels
unwarp_R = unwarped[:,:,0]
unwarp_G = unwarped[:,:,1]
unwarp_B = unwarped[:,:,2]
unwarp_HSV = cv2.cvtColor(unwarped, cv2.COLOR_RGB2HSV)
unwarp_H = unwarp_HSV[:,:,0]
unwarp_S = unwarp_HSV[:,:,1]
unwarp_V = unwarp_HSV[:,:,2]

fig, axs = plt.subplots(3,2, figsize=(16, 12))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
axs[0].imshow(unwarp_R, cmap='gray')
axs[0].set_title('RGB R-channel', fontsize=30)
axs[1].imshow(unwarp_G, cmap='gray')
axs[1].set_title('RGB G-Channel', fontsize=30)
axs[2].imshow(unwarp_B, cmap='gray')
axs[2].set_title('RGB B-channel', fontsize=30)
axs[3].imshow(unwarp_H, cmap='gray')
axs[3].set_title('HSV H-Channel', fontsize=30)
axs[4].imshow(unwarp_S, cmap='gray')
axs[4].set_title('HSV S-channel', fontsize=30)
axs[5].imshow(unwarp_V, cmap='gray')
axs[5].set_title('HSV V-Channel', fontsize=30)

fig.tight_layout()
# mpimg.imsave("test-after.jpg", color_select)
# plt.imsave("output_images/test_before2.jpg", img)
# plt.imsave("output_images/test_after2.jpg", undist)
plt.show()