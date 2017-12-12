#=============================================================================
#=== Importing libraries =====================================================
#=============================================================================

import numpy as np
import cv2
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
imageio.plugins.ffmpeg.download()
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#=============================================================================
#=== Reading the Saved Calibration Data and Original Image ===================
#=============================================================================

# Reads in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("calibration.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Reads an image
img = cv2.imread("./input_images/test1.jpg") 

#=============================================================================
#=== Undistorting Image ======================================================
#=============================================================================

def undistort(img):

	# Reads and makes a list of calibration images
	# images = glob.glob('./input_images/*.jpg')

	# fig, axs = plt.subplots(4, 2, figsize=(24, 10))
	# fig.tight_layout()
	# axs = axs.ravel()

	# for i, image in enumerate(images):
	# img = cv2.imread(image) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Undistorts the images as destination image
	undist = cv2.undistort(img, mtx, dist, None, mtx)

		# Visualizes undistortion one by one
	  # cv2.imshow('FRAME2',dst)
	  # cv2.waitKey(0)
	  # cv2.destroyAllWindows()

		# Visualizes undistortion in a 4x2 grid
	# 	axs[i].imshow(undist)
	# plt.show()

	print('Undistortion done!')
	return undist

#==========================================================================
#=== Perspective Transform ================================================
#==========================================================================

def perspecrive_transform(undist):
	
	img_size = (img.shape[1], img.shape[0])

	# Source points
	src = np.float32([(575,460),
	                  (707,460), 
	                  (260,680), 
	                  (1050,680)])

	# Destination points
	dst = np.float32([(450,0),
	                  (850,0),
	                  (450,700),
	                  (850,700)])

	# Calculates a perspective transform from four pairs of the .. 
	# corresponding points (quadrangle coordinates in the source ..
	# and destination image)
	PerspTrans = cv2.getPerspectiveTransform(src, dst)
	PerspTransInv = cv2.getPerspectiveTransform(dst, src)

	# fig2, axs = plt.subplots(4, 2, figsize=(24, 10))
	# fig2.tight_layout()
	# axs = axs.ravel()

	# for i, image in enumerate(images):
	# Applies a perspective transform to the image (front view to a top-down view)
	warped = cv2.warpPerspective(undist, PerspTrans, img_size, flags=cv2.INTER_LINEAR)

	# Visualizes undistortion one by one
	# cv2.imshow('FRAME2',dst)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# Visualizes undistortion in a 4x2 grid
	# plt.imshow(warped)
	# plt.plot(575,460,'.')
	# plt.plot(700,460,'.')
	# plt.plot(260,680,'.')
	# plt.plot(1050,680,'.')
	# # axs[i].imshow(warped)

	# plt.show()

	print('Perspective transform done!')  
	return warped, PerspTrans, PerspTransInv

#==========================================================================
#=== Gradient Threshold ===================================================
#==========================================================================

#===============================
#=== Calculates the gradient ===
#===============================
def abs_sobel_thresh(img, orient, thresh_min=25, thresh_max=255):

  # Converts the image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # Takes derivative in x or y with the cv2.sobel() function ..
  # and take the absolute value, we get the absolute gradient
  if orient == 'x':
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
  if orient == 'y':
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

  # Scales to 8-bit (0 - 255) then convert to 'np.uint8' type 
  scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

  # Creates a mask of 1's where the scaled gradient ..
  # is > thresh_min and < thresh_max
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
  
  print('Gradient threshold in x and y applied!')
  # Returns the binary image
  return binary_output

#=====================================
#=== Calculates gradient magnitude ===
#=====================================

def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):
  
  # Converts the image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # Takes gradient in x and y 
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

  # Calculates the gradient magnitude
  sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

  # Scales to 8-bit (0 - 255) then convert to 'np.uint8' type 
  scaledSobel = np.uint8(255*sobelxy/np.max(sobelxy))

  # Creates a binary mask where mag thresholds are met
  mag_binary = np.zeros_like(scaledSobel)
  mag_binary[(scaledSobel >= mag_thresh[0]) & (scaledSobel <= mag_thresh[1])] = 1

  print('Gradient magnitude threshold applied!')
  # Returns the binary image
  return mag_binary

#=====================================
#=== Calculates gradient direction ===
#=====================================

def dir_threshold(img, sobel_kernel=7, thresh=(0, 0.09)):

  # Converts the image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # Calculates the x and y gradients
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

  # Takes the absolute value of the x and y gradients
  abs_sobelx = np.absolute(sobelx)
  abs_sobely = np.absolute(sobely)

  # Uses np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
  gradDir = np.arctan2(abs_sobely, abs_sobelx)

  # Creates a binary mask where direction thresholds are met
  dir_binary = np.zeros_like(gradDir)
  dir_binary[(gradDir >= thresh[0]) & (gradDir <= thresh[1])] = 1

  print('Gradient direction threshold applied!')
  # Returns the binary image
  return dir_binary

#=======================================================================
#=== Color Space Threshold =============================================
#=======================================================================

def hls_threshold(img, h_thresh=(15, 100), l_thresh=(10, 150), s_thresh=(90, 255)):
    
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,1]
    H = H*(255/np.max(H))
    L = hls[:,:,0]
    L = L*(255/np.max(L))
    S = hls[:,:,2]
    S = S*(255/np.max(S))

    # Apply a threshold to the channels
    h_binary = np.zeros_like(H)
    h_binary[(H > h_thresh[0]) & (H <= h_thresh[1])] = 1

    l_binary = np.zeros_like(L)
    l_binary[(L > l_thresh[0]) & (L <= l_thresh[1])] = 1

    s_binary = np.zeros_like(S)
    s_binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1

    print('Color space threshold applied!')
    # Returns the binary image
    return h_binary, l_binary, s_binary

#==========================================================================
#=== Fitting a Polynomial to the Lane Lines ===============================
#==========================================================================

def initial_fit_polynomial(binary_warped):
	
	# Takes a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Creates an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Finds the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Chooses the number of sliding windows
	nwindows = 9

	# Sets height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)

	# Identifies the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Sets the width of the windows +/- margin
	margin = 100
	# Sets minimum number of pixels found to recenter window
	minpix = 50
	# Creates empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Steps through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
	    (0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
	    (0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenates the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extracts left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fits a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	print('Fitted the initial polynomial to the lane lines!')

	# Generates x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()

	print('Visualized the initial sliding windows and fitted a polynomial!')
	return left_fit, right_fit, left_lane_inds, right_lane_inds

#==========================================================================
#=== Fitting a Polynomial to the Lane Lines ===============================
#==========================================================================

def next_fit_polynomial(grad_and_color_combined, left_fit, right_fit):
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100

	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extracts left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fits a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	print('Fitted the polynomial to the lane lines!')

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	                              ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	                              ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	print('Visualized the sliding windows and fitted a polynomial!')
	return left_fit, right_fit, left_lane_inds, right_lane_inds

#==========================================================================
#=== Processing Image Pipeline ============================================
#==========================================================================

def process_image(img, count):
	new_img = np.copy(img)
	
	# Applies a distortion correction to raw images.
	undist = undistort(new_img)

	# Applies a perspective transform to rectify image ("birds-eye view")
	warped, PerspTrans, PerspTransInv = perspecrive_transform(undist)

  # Calculates gradients (magnitude and direction) and ..
  # applies them to create a thresholded binary image
	grad_binary_x = abs_sobel_thresh(warped, orient='x')
	grad_binary_y = abs_sobel_thresh(warped, orient='y')
	mag_binary = mag_thresh(warped)
	dir_binary = dir_threshold(warped)

  # Calculates HLS color space and ..
  # applies them to create a thresholded binary image 
	h_binary, l_binary, s_binary = hls_threshold(warped)

	gradient_combined = np.zeros_like(mag_binary)
	color_combined = np.zeros_like(l_binary)
	grad_and_color_combined = np.zeros_like(l_binary)

  # Combines the gradient binary thresholds
	gradient_combined[((grad_binary_x == 1) & (grad_binary_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
  
  # Combines the color space binary thresholds
	color_combined[((l_binary == 1) & (s_binary == 1) & (h_binary == 0))] = 1
  
  # Combines the combined gradient and combined color binary thresholds
	grad_and_color_combined[((gradient_combined == 1) | (color_combined == 1))] = 1
	print('Gradient and color space thresholds combined!')

	# if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
	if not count:
		left_fit, right_fit, left_lane_inds, right_lane_inds = initial_fit_polynomial(grad_and_color_combined)
		count = True
	else:
		left_fit, right_fit, left_lane_inds, right_lane_inds = next_fit_polynomial(grad_and_color_combined, left_fit, right_fit)
	
  # Visualizes undistortion one by one
	# cv2.imshow('FRAME',grad_and_color_combined)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

count = False
for image in range(0, 5):
	process_image(img, count)
	image += 1