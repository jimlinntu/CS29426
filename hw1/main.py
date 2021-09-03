import argparse
import numpy as np
import cv2
import math

from pathlib import Path

DEBUG = True
DEBUG_FOLDER = Path("./debug")
if DEBUG:
    DEBUG_FOLDER.mkdir(exist_ok=True)

def write_debug_img(name, img):
    if DEBUG:
        cv2.imwrite(str(DEBUG_FOLDER / name), img)

class Aligner():
    def __init__(self):
        pass

    # Sum of Squared Differences
    def ssd(self, img1, img2, mask):
        img1 = img1.astype(np.int32)
        img2 = img2.astype(np.int32)
        return np.sum(np.square(img1-img2) * mask)

    def norm(self, img):
        return np.sqrt(np.sum(img ** 2))

    def ncc(self, img1, img2, mask):
        # Ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2004/10/tr-2004-92.pdf
        img1 = img1.astype(np.int32)
        img2 = img2.astype(np.int32)
        masked1 = img1[mask]
        masked2 = img2[mask]
        mean1 = np.mean(masked1)
        mean2 = np.mean(masked2)

        masked1 = masked1 - mean1
        masked2 = masked2 - mean2

        return np.sum((masked1 / self.norm(masked1)) * (masked2 / self.norm(masked2)))

    def shift(self, img, dx, dy):
        shifted = np.roll(img, (dy, dx), (0, 1))
        return shifted

    def img_grad(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        dimg_dx = cv2.Sobel(blur, cv2.CV_8U, 1, 0)
        dimg_dy = cv2.Sobel(blur, cv2.CV_8U, 0, 1)

        dimg_dx = dimg_dx.astype(np.int32)
        dimg_dy = dimg_dy.astype(np.int32)

        dimg = np.clip(np.sqrt(dimg_dx**2 + dimg_dy**2), 0, 255).astype(np.uint8)

        return dimg

    def align(self, img1, img2, x_bound, y_bound, metric):
        assert img1.shape == img2.shape
        assert isinstance(x_bound, tuple) and x_bound[0] < x_bound[1]
        assert isinstance(y_bound, tuple) and y_bound[0] < y_bound[1]

        h, w = img1.shape[0:2]

        # the region we want to compute the score
        mask = np.zeros((h, w), dtype=np.bool)
        # only consider the center region for the ssd
        rstart, rend = math.floor(h*0.1), math.floor(h*0.9)
        cstart, cend = math.floor(w*0.1), math.floor(w*0.9)
        mask[rstart:rend, cstart:cend] = True

        best_score = float("inf")
        best_dx, best_dy = 0, 0
        for dx in range(x_bound[0], x_bound[1]+1):
            for dy in range(y_bound[0], y_bound[1]+1):
                shifted = self.shift(img2, dx, dy)

                if metric == "ssd":
                    score = self.ssd(img1, shifted, mask)
                elif metric == "ncc":
                    # <= cos theta <= 1
                    score = self.ncc(img1, shifted, mask) * (-1)
                if score < best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy

        return (best_dx, best_dy)

    def multiscale_align(self, b, g, r, metric, depth=None):
        # decide the depth of the pyramid
        h, w = b.shape[0:2]
        if depth is None:
            depth = 0
            while w > 500:
                depth += 1
                w = w // 2

        print("Depth: {}".format(depth))

        b_pyramid = get_pyramid(b, depth)
        g_pyramid = get_pyramid(g, depth)
        r_pyramid = get_pyramid(r, depth)

        # multi scale alignment algorithm
        x_bounds = [(-2, 2) for i in range(depth)] + [(-20, 20)]
        y_bounds = [(-2, 2) for i in range(depth)] + [(-20, 20)]

        g_prev_dx, g_prev_dy = 0, 0
        r_prev_dx, r_prev_dy = 0, 0

        for i in range(depth, -1, -1):
            x_bound = x_bounds[i]
            y_bound = y_bounds[i]

            # because previous dx, dy is 2 times smaller
            g_prev_dx, g_prev_dy = g_prev_dx * 2, g_prev_dy * 2
            r_prev_dx, r_prev_dy = r_prev_dx * 2, r_prev_dy * 2

            b_i = b_pyramid[i]
            g_i = self.shift(g_pyramid[i], g_prev_dx, g_prev_dy)
            r_i = self.shift(r_pyramid[i], r_prev_dx, r_prev_dy)

            g_dx, g_dy = self.align(b_i, g_i, x_bound, y_bound, metric)
            r_dx, r_dy = self.align(b_i, r_i, x_bound, y_bound, metric)

            g_prev_dx += g_dx
            g_prev_dy += g_dy

            r_prev_dx += r_dx
            r_prev_dy += r_dy

        return (g_prev_dx, g_prev_dy), (r_prev_dx, r_prev_dy)

    def singlescale_align(self, b, g, r, metric):
        (g_dx, g_dy), (r_dx, r_dy) = self.multiscale_align(b, g, r, metric, depth=0)
        return (g_dx, g_dy), (r_dx, r_dy)

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

# resize to almost the same size
def resize(img, max_w=380):
    h, w = img.shape[0:2]
    scale = 1. if w <= max_w else max_w / w
    new_h = int(h * scale)
    new_w = max_w
    return new_h, new_w, scale

def get_pyramid(img, depth):
    assert isinstance(img, np.ndarray) and len(img.shape) == 2
    assert isinstance(depth, int)

    pyramid = [img]
    prev_img = img

    for i in range(depth):
        h, w = prev_img.shape
        next_img = cv2.resize(prev_img, (w // 2, h // 2))
        pyramid.append(next_img)
        prev_img = next_img

    return pyramid

def preprocess_raw_img(img):
    '''
        Automatic white boundary cropping
    '''
    # make it all w
    new_h, new_w, scale = resize(img)
    resized = cv2.resize(img, (new_w, new_h))
    # corner detection
    # Ref:
    # https://stackoverflow.com/questions/54720646/what-does-ksize-and-k-mean-in-cornerharris/54721585
    corner = cv2.cornerHarris(resized[:, :, 0], 3, 3, 0.06)
    corner_mask = (corner > corner.max() * 0.03).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(corner_mask)

    # coordinate transformation
    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

    img = img[y:y+h, x:x+w, ...]

    write_debug_img("pre_crop.jpg", img)
    write_debug_img("corner.jpg", corner_mask)
    return img

def crop_aligned(img):
    h, w = img.shape[0:2]
    rate = 0.02
    y_low, y_high = int(h * rate), int(h * (1-rate))
    x_low, x_high = int(w * rate), int(w * (1-rate))

    return img[y_low: y_high, x_low: x_high]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("algorithm", type=str, choices=["single", "multiscale", "mine"])
    parser.add_argument("metric", type=str, choices=["ssd", "ncc"])
    args = parser.parse_args()

    img = cv2.imread(args.img_path).astype(np.uint8)
    img = preprocess_raw_img(img)

    b, g, r = split(img)

    # test simply stack three channels
    if DEBUG:
        stacked = merge(b, g, r)
        write_debug_img("debug.jpg", stacked)

    aligner = Aligner()

    if args.algorithm == "single":
        (g_dx, g_dy), (r_dx, r_dy) = aligner.singlescale_align(b, g, r, args.metric)
    elif args.algorithm == "multiscale":
        (g_dx, g_dy), (r_dx, r_dy) = aligner.multiscale_align(b, g, r, args.metric)

    g = aligner.shift(g, g_dx, g_dy)
    r = aligner.shift(r, r_dx, r_dy)

    print("g_dx: {}, g_dy: {}".format(g_dx, g_dy))
    print("r_dx: {}, r_dy: {}".format(r_dx, r_dy))

    merged = merge(b, g, r)

    # result = crop_aligned(merged)

    cv2.imwrite("result.jpg", merged)



if __name__ == "__main__":
    main()
