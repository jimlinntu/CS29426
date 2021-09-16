import numpy as np
import cv2
from align_image_code import align_images, match_img_size

# import matplotlib.pyplot as plt

import argparse
from pathlib import Path

DEBUG = True

def freq_amplitude(img):
    log_amplitude = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    return log_amplitude

def get2d_gaussian(ksize, sigma):
    g1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = g1d.dot(g1d.T)
    return kernel

def get_lowpass_kernel(ksize, sigma):
    return get2d_gaussian(ksize, sigma)

def get_highpass_kernel(ksize, sigma, alpha):
    e = np.zeros((ksize, ksize), dtype=np.float32)
    mid = ksize // 2
    e[mid, mid] = 1
    highpass_kernel = (1 + alpha) * e - alpha * get2d_gaussian(ksize, sigma)
    return highpass_kernel

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    parser.add_argument("lp_k", type=int, help="low pass gaussian kernel size")
    parser.add_argument("hp_k", type=int, help="high pass laplacian kernel size")
    parser.add_argument("alpha", type=float, help="high pass laplacian's alpha")
    args = parser.parse_args()

    img1_stem = Path(args.img1).stem
    img2_stem = Path(args.img2).stem

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)

    img1, img2 = match_img_size(img1, img2)
    if DEBUG:
        print(img1.shape)
        print(img2.shape)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    amp1 = freq_amplitude(img1)
    amp1 = cv2.normalize(amp1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("{}_amp1.jpg".format(img1_stem), np.clip(amp1, 0, 255))

    amp2 = freq_amplitude(img2)
    amp2 = cv2.normalize(amp2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("{}_amp2.jpg".format(img2_stem), np.clip(amp2, 0, 255))

    # G1 = get_lowpass_kernel(51, 0)
    G1 = get_lowpass_kernel(args.lp_k, 0)
    G2 = get_highpass_kernel(args.hp_k, 0, args.alpha)

    img1_G1 = cv2.filter2D(img1, cv2.CV_32F, G1)
    img1_G1 = cv2.normalize(img1_G1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("{}_G1.jpg".format(img1_stem), img1_G1)

    img2_G2 = cv2.filter2D(img2, cv2.CV_32F, G2)
    img2_G2 = cv2.normalize(img2_G2, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("{}_G2.jpg".format(img2_stem), img2_G2)

    hybrid = (img1_G1 + img2_G2) / 2
    cv2.imwrite("{}_hybrid.jpg".format(img1_stem + "_" + img2_stem), hybrid)

if __name__ == "__main__":
    main()
