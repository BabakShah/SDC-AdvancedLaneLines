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
#=== Undistorting Image ======================================================
#=============================================================================

# Reads in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("calibration.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Reads and makes a list of calibration images
images = glob.glob('./input_images/*.jpg')

fig, axs = plt.subplots(4, 2, figsize=(24, 10))
fig.tight_layout()
axs = axs.ravel()

for i, image in enumerate(images):
	img = cv2.imread(image) 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Undistorts the images as destination image
	undist = cv2.undistort(img, mtx, dist, None, mtx)

	# Visualizes undistortion one by one
  # cv2.imshow('FRAME2',dst)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

	# Visualizes undistortion in a 4x2 grid
	axs[i].imshow(undist)
plt.show()

print('Undistortion done!')

#==========================================================================
#=== Perspective Transform ================================================
#==========================================================================

img_size = (img.shape[1], img.shape[0])

# Source points
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])

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

fig2, axs = plt.subplots(4, 2, figsize=(24, 10))
fig2.tight_layout()
axs = axs.ravel()

for i, image in enumerate(images):
	# Applies a perspective transform to the image (front view to a top-down view)
	warped = cv2.warpPerspective(undist, PerspTrans, img_size, flags=cv2.INTER_LINEAR)

	# Visualizes undistortion one by one
  # cv2.imshow('FRAME2',dst)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

	# Visualizes undistortion in a 4x2 grid
	plt.plot(575,460,'.')
	plt.plot(700,460,'.')
	plt.plot(260,680,'.')
	plt.plot(1050,680,'.')
	axs[i].imshow(warped)

plt.show()

print('Perspective transform done!')  