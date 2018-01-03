#=============================================================================
#=== Importing libraries =====================================================
#=============================================================================

import numpy as np
import os
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
img = cv2.imread("./input_images/test4.jpg") 
# img = cv2.imread("./Error_4.png") 

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
	
	img_size = (undist.shape[1], undist.shape[0])

	# Source points
	src = np.float32([(595,450),
	                  (685,450), 
	                  (265,720), 
	                  (1020,720)])

	# Destination points
	dst = np.float32([(450,0),
	                  (830,0),
	                  (450,700),
	                  (830,700)])

	# Calculates a perspective transform from four pairs of the .. 
	# corresponding points (quadrangle coordinates in the source ..
	# and destination image)
	PerspTrans = cv2.getPerspectiveTransform(src, dst)
	PerspTransInv = cv2.getPerspectiveTransform(dst, src)
	
	# for i, image in enumerate(images):
	# Applies a perspective transform to the image (front view to a top-down view)
	warped = cv2.warpPerspective(undist, PerspTrans, img_size, flags=cv2.INTER_LINEAR)

	# fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
	# fig2.tight_layout()
	# ax1.imshow(undist)
	# ax2.imshow(warped)

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
def abs_sobel_thresh(img, orient, thresh_min=10, thresh_max=150):

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

def mag_thresh(img, sobel_kernel=25, mag_thresh=(50, 100)):
  
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

def dir_threshold(img, sobel_kernel=25, thresh=(0.85, 1.15)):

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

def hls_threshold(img, h_thresh=(15, 100), l_thresh=(220, 255), s_thresh=(105, 205)):
    
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    H = H*(255/np.max(H))
    L = hls[:,:,1]
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
#=== Initial Fitting a Polynomial to the Lane Lines =======================
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

	# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	# plt.imshow(out_img)
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()

	print('Visualized the initial sliding windows and fitted a polynomial!')
	return left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds, ploty

#==========================================================================
#=== Fitting a Polynomial to the Lane Lines ===============================
#==========================================================================

def next_fit_polynomial(binary_warped, left_fit, right_fit):
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

	# Generates x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Creates an image to draw on and an image to show the selection window
	# out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# window_img = np.zeros_like(out_img)
	# # Color in left and right line pixels
	# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# # Generates a polygon to illustrate the search window area
	# # And recast the x and y points into usable format for cv2.fillPoly()
	# left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	# left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	#                               ploty])))])
	# left_line_pts = np.hstack((left_line_window1, left_line_window2))
	# right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	# right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	#                               ploty])))])
	# right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# # Draws the lane onto the warped blank image
	# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	# result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	# plt.imshow(result)
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()

	print('Visualized the sliding windows and fitted a polynomial!')
	return left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds, ploty

#==========================================================================
#=== Drawing The Lines Onto The Original Image ============================
#==========================================================================

def draw_lane(warped, undist, left_fitx, right_fitx, PerspTransInv, ploty):

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 255))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, PerspTransInv, (undist.shape[1], undist.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	# plt.imshow(result)
	# plt.show()

	print('Lane lines drawn on the image!')
	return result

def polynomial(line, value):

    a_coef = line[0]
    b_coef = line[1]
    c_coef = line[2]
    poly = (a_coef * value ** 2) + (b_coef * value) + c_coef

    return poly

#======================================================================
#=== radius of curvature and distance =================================
#======================================================================

# Calculates radius of curvature and distance from lane center 
def measure_curvature(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds, ploty):
  
  # Define conversions in x and y from pixels space to meters
  ym_per_pix = 30./720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension
  carm_pos = (1280 / 2) * xm_per_pix # lane center pos in meters

  left_curverad, right_curverad, center_dist = (0, 0, 0)

  # Define maximum y-value corresponding to the bottom of the image
  y_eval = np.max(ploty)

  # Identify the x and y positions of all nonzero pixels in the image
  nonzero = bin_img.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])

  leftx = nonzerox[l_lane_inds]
  lefty = nonzeroy[l_lane_inds] 
  rightx = nonzerox[r_lane_inds]
  righty = nonzeroy[r_lane_inds]
  
  # Fit new polynomials to x,y in world space
  left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  
  # Calculate the new radii of curvature
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

  left_line = polynomial(left_fit_cr, bin_img.shape[0] * ym_per_pix)
  right_line = polynomial(right_fit_cr, bin_img.shape[0] * ym_per_pix)
  center_dist = carm_pos - ((left_line + right_line) / 2)

  # Now our radius of curvature is in meters
  print('Left lane radius: ',left_curverad,'m ','Right lane radius: ',right_curverad,'m' , 'Center lane radius: ',center_dist)
  return left_curverad, right_curverad, center_dist

#======================================================================
#=== Draw Radius of Curvature =========================================
#======================================================================

def draw_data(img, curv_rad, center_rad):
    new_img = np.copy(img)
    font = cv2.FONT_HERSHEY_DUPLEX
    text1 = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm' 
    cv2.putText(new_img, text1, (40,70), font, 1.5, (0,255,255), 2, cv2.LINE_AA)
    text2 = 'Center distance: ' + '{:04.2f}'.format(center_rad) + 'm'
    cv2.putText(new_img, text2, (40,120), font, 1.5, (0,255,255), 2, cv2.LINE_AA)
    print('Lane data drawn on the image!')
    return new_img

#==========================================================================
#=== Processing Image Pipeline ============================================
#==========================================================================

def process_image(img):
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

	gradient_combined = np.zeros_like(grad_binary_x)
	color_combined = np.zeros_like(grad_binary_x)
	grad_and_color_combined = np.zeros_like(grad_binary_x)

	R_color = warped[:,:,0]
	r_thresh = (200, 255)
	r_binary = np.zeros_like(R_color)
	r_binary[(R_color > r_thresh[0]) & (R_color <= r_thresh[1])] = 1

  # Combines the gradient binary thresholds
	gradient_combined[((grad_binary_x == 1) & (grad_binary_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
	# gradient_combined[((grad_binary_x == 1) & (grad_binary_y == 1))] = 1
  # Combines the color space binary thresholds
	color_combined[((l_binary == 1) & (s_binary == 1) & (h_binary == 0))] = 1
  
  # Combines the combined gradient and combined color binary thresholds
	grad_and_color_combined[(((grad_binary_x == 1) & (s_binary == 1)) | ((grad_binary_x == 1) & (r_binary == 1)) | ((s_binary == 1) & (r_binary == 1)))] = 1
	# grad_and_color_combined[((grad_binary_x == 1) & (s_binary == 1) | (grad_binary_x == 1) & (l_binary == 1))] = 1
	print('Gradient and color space thresholds combined!')

	# fig2, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, figsize=(20, 10))
	# fig2.tight_layout()
	# ax1.imshow(undist)
	# ax2.imshow(warped)
	# ax3.imshow(grad_binary_x)
	# ax4.imshow(mag_binary)
	# ax5.imshow(r_binary)
	# ax6.imshow(s_binary)
	# ax7.imshow(grad_and_color_combined)
	# plt.show()

	# If it's not the first time fitting, uses next_fit_polynomial, otherwise uses initial_fit_polynomial
	if not l_line.detected or not r_line.detected:
		left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds, ploty = initial_fit_polynomial(grad_and_color_combined)

	else:
		left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds, ploty = next_fit_polynomial(grad_and_color_combined, left_fit, right_fit)
	
	img_with_lane = draw_lane(grad_and_color_combined, undist, left_fitx, right_fitx, PerspTransInv, ploty)
	
	rad_left, rad_right, rad_center = measure_curvature(grad_and_color_combined, left_fitx, right_fitx, left_lane_inds, right_lane_inds, ploty)
	img_with_lane_and_data = draw_data(img_with_lane, (rad_left+rad_right)/2, rad_center)
  
	processed_img = cv2.cvtColor(img_with_lane_and_data, cv2.COLOR_BGR2RGB)
  	# Visualizes the image with lane line fits and radius data
	# cv2.imshow('FRAME', processed_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	
	plt.imshow(img_with_lane_and_data)
	plt.show()

	return processed_img
#======================================================================
#=== Tracking Lane Lines ==============================================
#======================================================================

# Defines a class to receive the characteristics of each line detection
class Line():
  def __init__(self):
    # was the line detected in the last iteration?
    self.detected = False  
    # x values of the last n fits of the line
    self.recent_xfitted = [] 
    #average x values of the fitted line over the last n iterations
    self.bestx = None     
    #polynomial coefficients averaged over the last n iterations
    self.best_fit = None  
    #polynomial coefficients for the most recent fit
    # self.current_fit = [np.array([False])]  
    self.current_fit = []
    #radius of curvature of the line in some units
    self.radius_of_curvature = None 
    #distance in meters of vehicle center from the line
    self.line_base_pos = None 
    #difference in fit coefficients between last and new fits
    self.diffs = np.array([0,0,0], dtype='float') 
    #x values for detected line pixels
    self.allx = None  
    #y values for detected line pixels
    self.ally = None

l_line = Line()
r_line = Line()

for image in range(0, 2):
	process_image(img)
	image += 1

# input_video = VideoFileClip('project_video_test.mp4')
# output_video = input_video.fl_image(process_image)
# output_video = 
# .subclip(10, 15)
# print (video_input1.fps) 
# video_input2 = video_input1
# processed_video = video_input1.fl_image(process_image)

# output_video.write_videofile('project_video_test_output3.mp4', audio=False)
