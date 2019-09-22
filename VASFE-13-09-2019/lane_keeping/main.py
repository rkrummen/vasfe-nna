# Responsible for tracking the car inside a lane
# All variables must be kept in the local scope of main

from lane_keeping.poly_fit import main as poly_fit

def main(img, DEBUG):
    left_fit, right_fit = poly_fit(img, DEBUG)
    return left_fit, right_fit