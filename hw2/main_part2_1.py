import numpy as np
import cv2
import argparse

from pathlib import Path

DEBUG = False

def get2d_gaussian(ksize):
    g = cv2.getGaussianKernel(ksize, 0)
    kernel = g.dot(g.T)
    return kernel

def get_unit_impulse(ksize):
    kernel = np.zeros((ksize, ksize))
    mid = ksize // 2
    kernel[mid, mid] = 1
    return kernel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("alpha", type=float)
    parser.add_argument("out", type=str)
    parser.add_argument("--blur", default=False, action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if DEBUG:
        print(f"img.shape: {img.shape}")

    ksize = 5
    G = get2d_gaussian(ksize)
    e = get_unit_impulse(ksize)

    if args.blur:
        p = Path(args.image)
        parent = p.parent
        name = str(p.stem + "_blur" + p.suffix)
        blur_path = name
        blur = cv2.filter2D(img, cv2.CV_32F, G)
        img = np.clip(blur, 0, 255).astype(np.uint8)
        cv2.imwrite("{}".format(blur_path), img)
    alpha = args.alpha


    # Laplacian of Gaussian

    # alpha controls how much we want to sharpen this image
    # derived from
    # I + a * (I - I * G) = I *((1 + a)e - a * G)
    LoG = (1 + alpha) * e - alpha * G

    sharpened = cv2.filter2D(img, cv2.CV_32F, LoG)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{args.out}", sharpened)


if __name__ == "__main__":
    main()
