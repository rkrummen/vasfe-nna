# Main execution
import cv2
#from sensors.main import main as sensors
from pre_processing.main import main as pre_processing
import matplotlib.image as mpimg
from lane_keeping.main import main as lane_keeping

def main():
    #Img should be read from the camera and passed from sensors module
    #DEBUG=True shows img plots
    img = mpimg.imread('car_img_uem.jpg')
    processed_img = pre_processing(img, DEBUG=True)
    #Returns left and right lane fitted polynomials
    left_fit, right_fit = lane_keeping(processed_img, DEBUG=True)
    return 0

if __name__ == '__main__':
    main()