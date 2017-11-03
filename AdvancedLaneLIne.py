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

    objpoints = []
    imgpoints = []

    images = glob.glob('./camera_cal_images/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(100)

    cv2.destroyAllWindows()
    return objpoints, imgpoints


def calibration(objpoints, imgpoints):
    # Read an image
    img = cv2.imread('./camera_cal_images/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return mtx, dist

def undistortion(mtx, dist):
    img = cv2.imread('./camera_cal_images/calibration1.jpg')
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return img, undist


objpoints, imgpoints = points()


print(objpoints)
print(imgpoints)
mtx, dist = calibration(objpoints, imgpoints)


print("mtx: ",mtx)
print("dist: ",dist)
img, undist = undistortion(mtx, dist)

# Visualize undistortion
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

axs[0].imshow(img)
axs[0].set_title('Original Image', fontsize=30)

axs[1].imshow(undist)
axs[1].set_title('Undistorted Image', fontsize=30)

fig.tight_layout()
# mpimg.imsave("test-after.jpg", color_select)
# plt.imsave("test_output/test_after.jpg", results)
plt.show()