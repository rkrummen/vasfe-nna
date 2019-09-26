# Responsible for tracking the car inside a lane
# All variables must be kept in the local scope of main

from lane_keeping.poly_fit import main as poly_fit
from lane_keeping.find_curvature import main as find_curvature

def main(img, DEBUG):
    left_fit, right_fit, plot_y = poly_fit(img, DEBUG)
    left_curverad, right_curverad = find_curvature(left_fit, right_fit, plot_y, DEBUG)
    return left_fit, right_fit, left_curverad, right_curverad