import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def camera_calibration(img):
    # Load calibration data
    with np.load('pre_processing/calibration.npz') as data:
        mtx, dist = data['mtx'], data['dist']

    h, w = img.shape[:2]

    # Get camera matrix to handle different resolution
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return undistorted

def blur(img):
    # Applies Gaussian blur to reduce artifacts
    blur = cv2.GaussianBlur(img,(7,7),0)
    return blur

def hls(img):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    return s_channel

def x_threshold_gradient(img):
    # Sobel x
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 25
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def color_threshold(img):
    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(img)
    s_binary[(img >= s_thresh_min) & (img <= s_thresh_max)] = 1
    return s_binary

def perspective_warp(img):
    # Modify perspective (bird-eye view)
    pts1 = np.float32([[889, 561], [1192, 561], [267, 903], [1598, 903]])
    pts2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (1920, 1080))
    return dst

def main(img, DEBUG):
    # undistorted_img = camera_calibration(img)
    undistorted_img = img
    blur_img = blur(undistorted_img)
    hls_img = hls(blur_img)
    x_threshold_gradient_img = x_threshold_gradient(hls_img)
    color_threshold_img = color_threshold(hls_img)

    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(x_threshold_gradient_img), x_threshold_gradient_img, color_threshold_img)) * 255

    # Combine the two binary thresholds
    combined_binary_img = np.zeros_like(x_threshold_gradient_img)
    combined_binary_img[(color_threshold_img == 1) | (x_threshold_gradient_img == 1)] = 1

    perspective_warp_img = perspective_warp(combined_binary_img)

    if DEBUG == True:
        def save_image(data, cm, fn):
            #Saves undistored image (data=image, cm=colormap, fn=filename)
            sizes = np.shape(data)
            height = float(sizes[0])
            width = float(sizes[1])
            
            fig = plt.figure()
            fig.set_size_inches(width/height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        
            ax.imshow(data, cmap=cm)
            plt.savefig(fn, dpi = height) 
            plt.close()
        #save_image(undistorted_img, 'gray', 'undistorted_img.jpg') 

        # Plotting undistored image
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        ax1.set_title('Original image')
        ax1.imshow(img)

        ax2.set_title('Calibrated image')
        ax2.imshow(undistorted_img, cmap='gray')
        plt.show()

        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary_img, cmap='gray')
        plt.show()

        # Plotting processed input and perspective warped img
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        ax1.set_title('Input')
        ax1.imshow(combined_binary_img, cmap='gray')

        ax2.set_title('Output')
        ax2.imshow(perspective_warp_img, cmap='gray')
        plt.show()
    return perspective_warp_img

