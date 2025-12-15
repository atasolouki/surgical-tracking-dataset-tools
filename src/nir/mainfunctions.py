import cv2
import numpy as np

def load_image(pathname: str):
    """
    Wrapper for reading image from file path
    :param pathname: string to full image pathname
    :return: image as a numpy array
    """

    return cv2.imread(pathname)

def generate_timestamps_camera(pathname: str):
    """
    Generates timestamps file
    :param pathname: string to full pathname of directory containing all images in the recording
    :return: None
    """

    pass

def rescale(im, res_factor = 2.0):
    return cv2.resize(im,None,fx=res_factor,fy=res_factor,interpolation=cv2.INTER_AREA)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def build_log_and_gamma_table(gamma=1.0,beta=1.0):
    invGamma = 1.0 / gamma
    table = np.zeros(256,dtype=np.uint8)
    table[1:] = np.array([min(255,((np.log10(i) * beta/255.0) ** invGamma) * 255.0)
                      for i in np.arange(1, 256)]).astype("uint8") #log of 0 is undefined, keep at 0
    return table
