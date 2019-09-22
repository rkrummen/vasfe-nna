# Responsible for image pre-processing
# Returns an image, all variables must be kept in the local scope of main

from pre_processing.img_processing import main as pre_processing

def main(img, DEBUG):
    processed_img = pre_processing(img, DEBUG)
    return processed_img
