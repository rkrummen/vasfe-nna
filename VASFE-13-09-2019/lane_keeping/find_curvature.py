import numpy as np
import matplotlib.pyplot as plt
import cv2

def measure_curvature_real(left_fit, right_fit, plot_y):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 0.1 # meters per pixel in y dimension
    xm_per_pix = 0.1 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(plot_y)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

def main(left_fit, right_fit, plot_y, DEBUG):
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, plot_y)

    if DEBUG == True:
        print(left_curverad);
        print(right_curverad);
    
    return left_curverad, right_curverad