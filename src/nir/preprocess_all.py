import cv2
import glob
import numpy as np
import os

from pathlib import Path
from utils.mainfunctions import adjust_gamma, rescale, build_log_and_gamma_table

PATH = 'Raw_data/camera/20240418_123532_Phantom'
SAVEPATH = 'Enhanced/camera/20240418_123532_Phantom'
EXT = ['png', 'tif']

CAMS = ['DEV_1AB22C014CA5', 'DEV_1AB22C014CAF', 'DEV_1AB22C013857']
CAM_SEL = 1
FPS = 25.6

delta = 0
alpha = 1.00
gamma = 0.25
theta = 90.0

input_image = None
curr_lf = None
base_lf = None


def lf_div(image, lightfield, alpha=alpha, delta=delta):
    lf_factor = (np.divide(lightfield * (1 - alpha) + alpha, lightfield))

    image_flt = image.astype(np.float_) + delta
    image_div = image_flt * lf_factor[..., np.newaxis]
    # image_div = np.clip(image_div,0,255)
    return image_div


def process(image, gamma=gamma, theta=theta):
    # build_log_and_gamma_table(0.25, 120.0)

    image = (np.log10(image)) * theta
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return adjust_gamma(image, gamma)


def main():
    for cam in CAMS:

        folder = PATH
        imgs = sorted(f for f in glob.glob(f'{folder}/{cam}*') if EXT[0] in f or EXT[1] in f)
        success = len(imgs) > 0
        fnum = -1

        # Loop through the frames
        lf = cv2.imread(f'{cam}_LF.png', cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        lf = lf.astype(np.float32) / (2 ** 16 - 1)

        # frame_prev = None

        while success:
            # Read a frame from the video

            fnum = fnum + 1
            success = fnum < len(imgs)
            if not success:
                break

            if fnum % 1 != 0:
                continue

            path = imgs[fnum * 1]
            stem = Path(path).stem
            savepath = os.path.join(SAVEPATH, f'{stem}.png')

            frame = cv2.imread(imgs[fnum * 1])

            if fnum % 1 != 0:
                continue

            cv2.imshow("orig", rescale(frame, 0.5))
            frame1 = lf_div(frame, lf, alpha=alpha, delta=delta)
            frame1 = process(frame1, gamma=gamma, theta=theta)
            frame1 = cv2.GaussianBlur(frame1, (5, 5), sigmaX=2.0)

            cv2.imshow("frame", rescale(frame1, 0.5))
            cv2.imwrite(savepath,frame1)

            k = cv2.waitKey(1) & 255

            if k == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
