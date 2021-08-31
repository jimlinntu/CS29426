import argparse
import numpy as np
import cv2
import math

from pathlib import Path

DEBUG = True
DEBUG_FOLDER = Path("./debug")
if DEBUG:
    DEBUG_FOLDER.mkdir(exist_ok=True)

class Aligner():
    def __init__(self):
        pass

    # Sum of Squared Differences
    def ssd(self, img1, img2, mask):
        return np.sum(np.square(img1-img2) * mask)

    def shift(self, img, dx, dy):
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)
        return shifted

    def preprocess(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        dx = cv2.Sobel(blur, cv2.CV_8U, 1, 0)
        return dx

    # align img2 to img1
    def align(self, img1, img2):
        assert img1.shape == img2.shape
        h, w = img1.shape[0:2]
        x_bound = (-20, 20)
        y_bound = (-20, 20)


        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)

        # the region we want to compute the score
        mask = np.zeros((h, w))
        # only consider the center region
        rstart, rend = math.floor(h*0.2), math.floor(h*0.8)
        cstart, cend = math.floor(w*0.2), math.floor(w*0.8)
        mask[rstart:rend, cstart:cend] = 1

        best_score = float("inf")
        best_dx, best_dy = 0, 0
        for dx in range(x_bound[0], x_bound[1]+1):
            for dy in range(y_bound[0], y_bound[1]+1):
                shifted = self.shift(img2, dx, dy)

                score = self.ssd(img1, shifted, mask)
                if score < best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy

        return (best_dx, best_dy)

def split(img):
    assert isinstance(img, np.ndarray)
    H, w, _ = img.shape[:3]
    h = H // 3
    b = img[0:h, :, 0]
    g = img[h:2*h, :, 0] 
    # NOTE: when H != 3k. Ex. 7 -> 2 2 3 -> 2 2 2
    r = img[2*h:3*h, :, 0] 
    return b, g, r

def merge(b, g, r):
    merged = np.stack([b, g, r], axis=2)
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    args = parser.parse_args()

    img = cv2.imread(args.img_path)
    img = img.astype(np.uint8)
    b, g, r = split(img)

    if DEBUG:
        stacked = merge(b, g, r)
        cv2.imwrite(str(DEBUG_FOLDER / "debug.jpg"), stacked)

    aligner = Aligner()
    dx, dy = aligner.align(b, g)
    g = aligner.shift(g, dx, dy)
    print(dx, dy)
    dx, dy = aligner.align(b, r)
    r = aligner.shift(r, dx, dy)
    print(dx, dy)

    merged = merge(b, g, r)
    cv2.imwrite(str(DEBUG_FOLDER / "result.jpg"), merged)


if __name__ == "__main__":
    main()
