import argparse
import numpy as np
import cv2
import math

from pathlib import Path

DEBUG = False
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

    def align(self, img1, img2, x_bound, y_bound, metric, center_mask):
        assert img1.shape == img2.shape
        assert isinstance(x_bound, tuple) and x_bound[0] < x_bound[1]
        assert isinstance(y_bound, tuple) and y_bound[0] < y_bound[1]
        assert isinstance(center_mask, bool)

        h, w = img1.shape[0:2]

        # the region we want to compute the score
        mask = np.zeros((h, w), dtype=np.bool)
        # only consider the center region for the ssd
        if center_mask:
            rstart, rend = math.floor(h*0.1), math.floor(h*0.9)
            cstart, cend = math.floor(w*0.1), math.floor(w*0.9)
        else:
            rstart, rend = 0, h
            cstart, cend = 0, w

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

    def multiscale_align(self, base_img, img2, img3, metric, center_mask, depth=None):
        # decide the depth of the pyramid
        h, w = base_img.shape[0:2]
        if depth is None:
            depth = 0
            while w > 500:
                depth += 1
                w = w // 2

        print("Depth: {}".format(depth))

        base_pyramid = get_pyramid(base_img, depth)
        img2_pyramid = get_pyramid(img2, depth)
        img3_pyramid = get_pyramid(img3, depth)

        # multi scale alignment algorithm
        x_bounds = [(-2, 2) for i in range(depth)] + [(-20, 20)]
        y_bounds = [(-2, 2) for i in range(depth)] + [(-20, 20)]

        prev_dx1, prev_dy1 = 0, 0
        prev_dx2, prev_dy2 = 0, 0

        for i in range(depth, -1, -1):
            x_bound = x_bounds[i]
            y_bound = y_bounds[i]

            # because previous dx, dy is 2 times smaller
            prev_dx1, prev_dy1 = prev_dx1 * 2, prev_dy1 * 2
            prev_dx2, prev_dy2 = prev_dx2 * 2, prev_dy2 * 2

            b_i = base_pyramid[i]
            img2_i = self.shift(img2_pyramid[i], prev_dx1, prev_dy1)
            img3_i = self.shift(img3_pyramid[i], prev_dx2, prev_dy2)

            dx1, dy1 = self.align(b_i, img2_i, x_bound, y_bound, metric, center_mask)
            dx2, dy2 = self.align(b_i, img3_i, x_bound, y_bound, metric, center_mask)

            prev_dx1 += dx1
            prev_dy1 += dy1

            prev_dx2 += dx2
            prev_dy2 += dy2

        return (prev_dx1, prev_dy1), (prev_dx2, prev_dy2)

    def singlescale_align(self, base_img, img2, img3, metric, center_mask):
        (dx1, dy1), (dx2, dy2) = self.multiscale_align(base_img, img2, img3, metric, center_mask, depth=0)
        return (dx1, dy1), (dx2, dy2)

    def img_grad(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        dx = cv2.Sobel(blurred, cv2.CV_8UC1, 1, 0)
        dy = cv2.Sobel(blurred, cv2.CV_8UC1, 0, 1)

        dx = dx.astype(np.int32)
        dy = dy.astype(np.int32)

        grad_len = np.sqrt(dx ** 2 + dy ** 2)
        return grad_len

    def my_align(self, img1, img2, x_bound, y_bound):
        h, w = img1.shape[0:2]

        img1 = self.img_grad(img1)
        img2 = self.img_grad(img2)

        # the region we want to compute the score
        mask = np.zeros((h, w), dtype=np.int32)
        # only consider the center region for the ssd
        rstart, rend = math.floor(h*0.1), math.floor(h*0.9)
        cstart, cend = math.floor(w*0.1), math.floor(w*0.9)

        mask[rstart:rend, cstart:cend] = True

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

    def my_multiscale_align(self, base_img, img2, img3, depth=None):
        # decide the depth of the pyramid
        h, w = base_img.shape[0:2]
        if depth is None:
            depth = 0
            while w > 500:
                depth += 1
                w = w // 2

        print("Depth: {}".format(depth))

        base_pyramid = get_pyramid(base_img, depth)
        img2_pyramid = get_pyramid(img2, depth)
        img3_pyramid = get_pyramid(img3, depth)

        # multi scale alignment algorithm
        x_bounds = [(-2, 2) for i in range(depth)] + [(-20, 20)]
        y_bounds = [(-2, 2) for i in range(depth)] + [(-20, 20)]

        prev_dx1, prev_dy1 = 0, 0
        prev_dx2, prev_dy2 = 0, 0

        for i in range(depth, -1, -1):
            x_bound = x_bounds[i]
            y_bound = y_bounds[i]

            # because previous dx, dy is 2 times smaller
            prev_dx1, prev_dy1 = prev_dx1 * 2, prev_dy1 * 2
            prev_dx2, prev_dy2 = prev_dx2 * 2, prev_dy2 * 2

            b_i = base_pyramid[i]
            img2_i = self.shift(img2_pyramid[i], prev_dx1, prev_dy1)
            img3_i = self.shift(img3_pyramid[i], prev_dx2, prev_dy2)

            dx1, dy1 = self.my_align(b_i, img2_i, x_bound, y_bound)
            dx2, dy2 = self.my_align(b_i, img3_i, x_bound, y_bound)

            prev_dx1 += dx1
            prev_dy1 += dy1

            prev_dx2 += dx2
            prev_dy2 += dy2

        return (prev_dx1, prev_dy1), (prev_dx2, prev_dy2)

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

def color_balance(img, method):
    '''
        Balance this img's V channel by histogram equalization
        # https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
    '''
    h, w = img.shape[0:2]
    if method == "grey_world":
        mask = np.zeros((h, w), dtype=np.bool)
        mask[int(0.1*h): int(0.9*h), int(0.1*w):int(0.9*w)] = True
        crop_img = img[mask]

        b_mean = np.mean(crop_img[:, 0])
        g_mean = np.mean(crop_img[:, 1])
        r_mean = np.mean(crop_img[:, 2])

        gray = (b_mean + g_mean + r_mean) / 3

        b_factor = gray / b_mean
        g_factor = gray / g_mean
        r_factor = gray / r_mean

        new_img = img.copy().astype(np.float32)

        new_img[:, :, 0] *= b_factor
        new_img[:, :, 1] *= g_factor
        new_img[:, :, 2] *= r_factor
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)
        return new_img


    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    write_debug_img("v.jpg", y)

    if method == "histeq":
        y = cv2.equalizeHist(y)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(3, 3))
        y = clahe.apply(y)

    new_hsv = np.concatenate([np.expand_dims(y, axis=2), ycrcb[:, :, 1:3]], axis=2)
    new_img = cv2.cvtColor(new_hsv, cv2.COLOR_YCrCb2BGR)
    return new_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("base_channel", type=str, choices=["b", "g", "r"])
    parser.add_argument("algorithm", type=str, choices=["single", "multiscale", "mine"])
    parser.add_argument("metric", type=str, choices=["ssd", "ncc"])
    parser.add_argument("balance", type=str, choices=["none", "histeq", "clahe", "grey_world"])
    parser.add_argument("result", type=str)
    parser.add_argument("--no_corner_crop", default=False, action="store_true")
    parser.add_argument("--center_mask", default=False, action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.img_path).astype(np.uint8)
    if not args.no_corner_crop:
        img = preprocess_raw_img(img)

    b, g, r = split(img)

    # test simply stack three channels
    if DEBUG:
        stacked = merge(b, g, r)
        write_debug_img("debug.jpg", stacked)

    aligner = Aligner()

    base_img = b
    to_align = [g, r]
    if args.base_channel == "g":
        base_img = g
        to_align = [b, r]
    elif args.base_channel == "r":
        base_img = r
        to_align = [b, g]

    print("Using metric: {}".format(args.metric))
    print("Using center mask?: {}".format(args.center_mask))
    if args.algorithm == "single":
        (dx1, dy1), (dx2, dy2) = aligner.singlescale_align(base_img, to_align[0], to_align[1], args.metric, args.center_mask)
    elif args.algorithm == "multiscale":
        (dx1, dy1), (dx2, dy2) = aligner.multiscale_align(base_img, to_align[0], to_align[1], args.metric, args.center_mask)
    elif args.algorithm == "mine":
        (dx1, dy1), (dx2, dy2) = aligner.my_multiscale_align(base_img, to_align[0], to_align[1])


    img2 = aligner.shift(to_align[0], dx1, dy1)
    img3 = aligner.shift(to_align[1], dx2, dy2)

    if args.base_channel == "b":
        print("G: (dx, dy) = ({}, {}), R: (dx, dy) = ({}, {})".format(dx1, dy1, dx2, dy2))
    elif args.base_channel == "g":
        print("B: (dx, dy) = ({}, {}), R: (dx, dy) = ({}, {})".format(dx1, dy1, dx2, dy2))
    elif args.base_channel == "r":
        print("B: (dx, dy) = ({}, {}), G: (dx, dy) = ({}, {})".format(dx1, dy1, dx2, dy2))

    if args.base_channel == "b":
        merged = merge(base_img, img2, img3)
    elif args.base_channel == "g":
        merged = merge(img2, base_img, img3)
    elif args.base_channel == "r":
        merged = merge(img2, img3, base_img)

    if args.balance != "none":
        print("Using color balance method: {}".format(args.balance))
        merged = color_balance(merged, args.balance)

    cv2.imwrite(args.result + ".jpg", merged)

if __name__ == "__main__":
    main()
