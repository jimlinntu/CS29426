import cv2
import numpy as np
from tqdm import tqdm
import numba as nb

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import math
import argparse
from pathlib import Path
import re
from multiprocessing import Pool

class GooglePixel4a():
    '''
        The camera parameters for Google Pixel 4a
        1. https://store.google.com/us/product/pixel_4a_specs?hl=en-US
        2. https://capturetheatlas.com/camera-sensor-size/
    '''
    def __init__(self):
        # This refers to the diagonal field of view
        # NOTE: one interesting here is:
        # I originally though this FOV is referring to horizontal fov (hfov)
        # turns out after trial and error, I found out it is actually diagonal fov(dfov)
        # https://learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
        self.diag_fov = 77
        self.pixel_in_mm = 1.4e-3 # 1.4um

        self.w_in_pixel = 4032
        self.h_in_pixel = 3024
        self.diag_in_pixel = math.sqrt(self.w_in_pixel**2 + self.h_in_pixel**2)

    def diag_in_mm(self):
        return self.diag_in_pixel * self.pixel_in_mm

    # get_f_in_mm will return 4.43 mm which align with the EXIF 4.38mm
    def get_f_in_mm(self):
        '''
            Solve:
                tan(theta/2) = w / 2f
            =>  2f = w / tan(theta/2)
            =>   f = w / 2*tan(theta/2)
        '''
        return (1/2) * self.diag_in_mm() / math.tan(math.radians(self.diag_fov/2))

    def get_f_in_pixels(self, rounded=True):
        r = (self.get_f_in_mm() / self.pixel_in_mm)
        return round(r)

class Projector():
    def __init__(self, f):
        self.f = int(f)

    def img_to_cylinder(self, points):
        # (n, 2)
        assert len(points.shape) == 2
        assert points.shape[1] == 2 # (x, y)

        # (n, )
        x = points[:, 0]
        y = points[:, 1]

        # the distance to camera
        # (n, )
        d = np.sqrt(x**2 + self.f**2)

        # (x, y, f) -> (fx/d, f y/d, f^2/d)
        x_cylinder = self.f * x  / d
        y_cylinder = self.f * y / d
        z_cylinder = self.f * self.f / d
        cylinder_points = np.stack([x_cylinder, y_cylinder, z_cylinder])

        # Convert the cylinder points to its u, v coordinates
        # theta = atan(x, f)
        # u = f theta
        # v = y_clinder
        theta = np.arctan2(x, self.f)
        u = self.f * theta
        v = y_cylinder

        # (n, 2)
        return np.stack([u, v], axis=1)

    # points on the uv (cylinder coordinates)
    def cylinder_to_img(self, points):
        assert points.shape[1] == 2
        u = points[:, 0]
        v = points[:, 1]

        x = self.f * np.tan(u/self.f)
        y = v * np.sqrt(x**2 + self.f**2) / self.f

        # (n, 2) (ignore the last coordinates f)
        # i.e. (x, y, f)
        #           ^^^^ ignore this coordinates
        return np.stack([x, y], axis=-1)


def load_jpg_imgs_from_folder(folder):
    folder = Path(folder)
    pattern = re.compile("\.jpg$")
    files = [p for p in folder.iterdir() if pattern.search(p.name)]

    # Sort according to its time tags
    files.sort(key=lambda p: p.name)

    imgs = []
    for p in files:
        img = cv2.imread(str(p))
        imgs.append(img)
    return imgs

def test():
    img = cv2.imread("./src_imgs/Checkerboard.png")

    h, w = img.shape[0:2]
    boundaries = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1],
                           [w//2, 0], [w//2, h-1]])

    center = np.mean(boundaries, axis=0, keepdims=True).astype(np.int32)

    projector = Projector(800)
    # (n, 2)
    uv = projector.img_to_cylinder(boundaries - center)
    uv = uv.astype(np.int32)

    top_left_uv = np.min(uv, axis=0)
    bot_right_uv = np.max(uv, axis=0)

    cylinder_w, cylinder_h = (bot_right_uv - top_left_uv + 1)

    u_range = np.arange(0, cylinder_w) + top_left_uv[0]
    v_range = np.arange(0, cylinder_h) + top_left_uv[1]

    # (cylinder_h, cylinder_w, 2)
    U, V = np.meshgrid(u_range, v_range)

    uv_mesh = np.stack([U, V], axis=-1)

    # Each pixel's (u, v) -> (x, y)
    map_ = projector.cylinder_to_img(uv_mesh.reshape(-1, 2))

    # (cylinder_h, cylinder_w, 2)
    map_ = map_.reshape(cylinder_h, cylinder_w, 2)

    map_ = map_ + center[np.newaxis, :, :]

    map_ = map_.astype(np.float32)

    out = cv2.remap(img, map_[:, :, 0], map_[:, :, 1], cv2.INTER_LINEAR)

    cv2.imwrite("out.jpg", out)

def make_cylinder_img(img, f):
    h, w = img.shape[0:2]
    boundaries = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1],
                           [w//2, 0], [w//2, h-1]])

    # (1, 2)
    center = np.array([[w//2, h//2]], dtype=np.int32)

    projector = Projector(f)
    # (n, 2)
    uv_boundaries = projector.img_to_cylinder(boundaries - center)
    uv_boundaries = uv_boundaries.astype(np.int32)

    top_left_uv = np.min(uv_boundaries, axis=0)
    bot_right_uv = np.max(uv_boundaries, axis=0)

    # Determine the uv image size
    cylinder_w, cylinder_h = (bot_right_uv - top_left_uv + 1)

    # The image is in [top_left_uv[0], bot_right_uv[0]], [top_left_uv[1], bot_right_uv[1]]
    u_range = np.arange(top_left_uv[0], bot_right_uv[0]+1)
    v_range = np.arange(top_left_uv[1], bot_right_uv[1]+1)

    # (cylinder_h, cylinder_w, 2)
    U, V = np.meshgrid(u_range, v_range)

    uv_mesh = np.stack([U, V], axis=-1)

    # Each pixel's (u, v) -> (x, y)
    map_ = projector.cylinder_to_img(uv_mesh.reshape(-1, 2))

    # (cylinder_h, cylinder_w, 2)
    map_ = map_.reshape(cylinder_h, cylinder_w, 2)

    map_ = map_ + center[np.newaxis, :, :]

    map_ = map_.astype(np.float32)

    out = cv2.remap(img, map_[:, :, 0], map_[:, :, 1], cv2.INTER_LINEAR)
    return out

def get_pyramid(img, depth):
    prev_img = img
    dtype = img.dtype
    pyramid = [img]
    for i in range(depth-1):
        prev_img = cv2.resize(prev_img.astype(np.float32), (prev_img.shape[1]//2, prev_img.shape[0]//2))
        pyramid.append(prev_img.astype(dtype))
    return pyramid

@nb.jit(nopython=True)
def shift(img, dx, dy):
    shifted = np.zeros_like(img, dtype=np.int32)

    h, w = img.shape[0:2]
    top_left = np.zeros((2, ), dtype=np.int32)
    bot_right = np.array([w-1, h-1], dtype=np.int32)

    shift_vec = np.array([dx, dy], dtype=np.int32)
    shift_top_left = top_left + shift_vec
    shift_bot_right = bot_right + shift_vec

    # Compute the intersection
    inter_top_left = np.maximum(top_left, shift_top_left)
    inter_bot_right = np.minimum(bot_right, shift_bot_right)

    new_w, new_h = (inter_bot_right - inter_top_left + 1)

    # the index to the original image
    x, y = inter_top_left - shift_top_left
    shifted[inter_top_left[1]:inter_top_left[1]+new_h, inter_top_left[0]:inter_top_left[0]+new_w, :] =\
        img[y:y+new_h, x:x+new_w, :]
    return shifted

def fast_compute_offset(
        f_pyramid, f_pyramid_mask,
        m_pyramid, m_pyramid_mask):
    # Pyramid search
    first_search_range = (0, 500)
    search_range = (-30, 30) # only allow right shift because the way I took the photo
    prev_dx, prev_dy  = 0, 0

    depth = len(f_pyramid)
    for d in range(depth-1, -1, -1):
        f, f_mask = f_pyramid[d], f_pyramid_mask[d]
        m, m_mask = m_pyramid[d], m_pyramid_mask[d]

        best_dx, best_dy = 0, 0
        best_ssd = np.inf

        if d == depth-1:
            # The first time we search we know the image must be shifted right
            # that's why the first's search will be different from other layers
            min_dx, max_dx = first_search_range[0], first_search_range[1]
        else:
            min_dx, max_dx = search_range[0] // (depth-d), search_range[1] // (depth-d)

        if d == depth-1:
            # min_dy, max_dy = -1, 1
            min_dy, max_dy = -10, 10
        else:
            min_dy, max_dy = -2, 2

        for dy in range(min_dy, max_dy+1):
            for dx in range(min_dx, max_dx+1):
                true_dx, true_dy = 2*prev_dx + dx, 2*prev_dy + dy
                # m_shifted = shift(m, dx, 0)
                # m_mask_shifted = shift(m_mask, dx, 0)

                m_shifted = np.roll(m, (true_dx, true_dy), axis=(1, 0))
                m_mask_shifted = np.roll(m_mask, (true_dx, true_dy), axis=(1, 0))
                if dx > 0:
                    m_mask_shifted[:, 0:dx, :] = 0 # this part should be zero
                elif dx < 0:
                    m_mask_shifted[:, dx:, :] = 0
                if dy > 0:
                    m_mask_shifted[0:dy, :, :] = 0
                elif dy < 0:
                    m_mask_shifted[dy:, :, :] = 0
                # Compute SSD
                intersection = (f_mask > 0) & (m_mask_shifted > 0)
                count_inter = np.sum(intersection)
                if count_inter == 0:
                    continue
                ssd = np.sum(np.square((f - m_shifted) * intersection)) / count_inter

                if ssd < best_ssd:
                    best_dx, best_dy = dx, dy
                    best_ssd = ssd

        # Update the guess and proceed to the next depth
        prev_dx = 2*prev_dx + best_dx
        prev_dy = 2*prev_dy + best_dy

    return np.array([prev_dx, prev_dy], dtype=np.int32)

def compute_offset(fixed_img, fixed_mask, move_img, move_mask):
    '''
        Find the best x s.t.
        after making move_img moved by that x,
        fixed_img and move_img can be pefectly aligned
    '''
    assert fixed_img.dtype == np.int32
    assert move_img.dtype == np.int32

    # Generate pyramids
    depth = 4
    f_pyramid = get_pyramid(fixed_img, depth)
    f_pyramid_mask = get_pyramid(fixed_mask, depth)

    m_pyramid = get_pyramid(move_img, depth)
    m_pyramid_mask = get_pyramid(move_mask, depth)

    return fast_compute_offset(f_pyramid, f_pyramid_mask, m_pyramid, m_pyramid_mask)

def get_dist_mask(mask):
    assert isinstance(mask, np.ndarray)
    assert len(mask.shape) == 3
    h, w = mask.shape[0:2]
    hh, ww = h+2, w+2
    # put 0 on the borders
    padded_mask = np.zeros((hh, ww), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask[:, :, 0]
    dst = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 3)
    dst = dst[1:-1, 1:-1, np.newaxis]
    dst = np.tile(dst, (1, 1, 3))
    return dst

class CylinderImage():
    def __init__(self, uv_top_left, cyl_img, cyl_mask):
        assert uv_top_left.shape == (2, ) and uv_top_left.dtype == np.int32
        assert isinstance(cyl_img, np.ndarray)
        assert isinstance(cyl_mask, np.ndarray)
        self.uv_top_left = uv_top_left
        self.cyl_img = cyl_img
        self.cyl_mask = cyl_mask
        self.h, self.w = self.cyl_img.shape[0:2]

    def bot_right(self):
        return self.uv_top_left + np.array([self.w-1, self.h-1], dtype=np.int32)

    def center(self):
        c = np.mean([self.uv_top_left, self.bot_right()], axis=0)
        return c.astype(np.int32)

    # Merge two images
    def merge(self, image):
        assert isinstance(image, CylinderImage)

        new_top_left = np.minimum(self.uv_top_left, image.uv_top_left)
        new_bot_right = np.maximum(self.bot_right(), image.bot_right())
        new_w, new_h = (new_bot_right - new_top_left + 1)

        img1 = np.zeros((new_h, new_w, 3), dtype=np.float32)
        mask1 = np.zeros((new_h, new_w, 3), dtype=np.float32)
        img2 = np.zeros((new_h, new_w, 3), dtype=np.float32)
        mask2 = np.zeros((new_h, new_w, 3), dtype=np.float32)

        tl = self.uv_top_left - new_top_left
        img1[tl[1]:tl[1]+self.h, tl[0]:tl[0]+self.w, :] = self.cyl_img
        mask1[tl[1]:tl[1]+self.h, tl[0]:tl[0]+self.w, :] = self.cyl_mask

        tl = image.uv_top_left - new_top_left
        img2[tl[1]:tl[1]+image.h, tl[0]:tl[0]+image.w, :] = image.cyl_img
        mask2[tl[1]:tl[1]+image.h, tl[0]:tl[0]+image.w, :] = image.cyl_mask

        out = self.weighted_sum(img1, img2, mask1, mask2)
        return CylinderImage(new_top_left, out, mask1+mask2)

    def weighted_sum(self, img1, img2, mask1, mask2):
        assert img1.shape == img2.shape == mask1.shape == mask2.shape

        inter = (mask1 > 0) & (mask2 > 0)
        only1 = (mask1 > 0) ^ inter
        only2 = (mask2 > 0) ^ inter

        out = np.zeros_like(img1)
        # Compute the overlapped region
        out[inter] = (mask1[inter] * img1[inter] + mask2[inter] * img2[inter]) / (mask1[inter] + mask2[inter])
        # Compute the region only in image 1
        out[only1] = img1[only1]
        # Compute the region only in image 2
        out[only2] = img2[only2]

        out = np.clip(out, 0, 255).astype(np.int32)
        return out

    def wrap_boundary(self, f, center_u, to_wrap):
        center_u = int(center_u)
        circum = int(round(f * math.radians(360)))

        # No need to wrap
        if self.w <= circum:
            return self.cyl_img

        assert self.w <= 2*circum, "You shouldn't provide images >= 360*2 degrees"

        # Put everything with u >= cyl_circum to u - cyl_circum

        # (h, circum, 3)
        out = self.cyl_img[:, 0:circum, :].copy()
        mask = self.cyl_mask[:, 0:circum, :]

        wrap = self.cyl_img[:, circum:, :]
        mask_wrap = self.cyl_mask[:, circum:, :].copy()
        if not to_wrap:
            # Throw away boundaries
            mask_wrap[:, :, :] = 0

        wrap_w = wrap.shape[1]

        # Weighted sum over the wrap region
        out[:, :wrap_w, :] =\
                self.weighted_sum(out[:, :wrap_w, :], wrap, mask[:, :wrap_w, :], mask_wrap)

        # out.shape[1] == circum
        # (2, )
        cur_center = self.center()
        # We want to change the center to center_u
        du = cur_center[0] - center_u
        out = np.roll(out, du, axis=1)
        return out

    def write(self, p, f, center_u, wrap):
        wrap = self.wrap_boundary(f, center_u, wrap)
        cv2.imwrite(p, wrap, [cv2.IMWRITE_JPEG_QUALITY, 20])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("center_idx", type=int)
    parser.add_argument("out", type=str)
    parser.add_argument("--wrap", default=False, action="store_true")
    args = parser.parse_args()

    if False:
        # test the cylinder mapping on a checkboard
        test()

    imgs = load_jpg_imgs_from_folder(args.folder)
    assert 0 <= args.center_idx < len(imgs)

    # Put this image to roughly in the middle of the list
    imgs = np.roll(imgs, len(imgs)//2-args.center_idx, axis=0)

    pixel4a = GooglePixel4a()

    for img in imgs:
        assert img.shape[0:2] == (pixel4a.h_in_pixel, pixel4a.w_in_pixel), "This image is not from Google Pixel 4a"

    cyl_imgs = []

    print("[*] Map each image to the cylinder")
    mask = np.ones(img.shape, dtype=np.uint8)
    mask = get_dist_mask(mask)
    if False:
        cv2.imwrite("mask.jpg", cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))

    cyl_mask = make_cylinder_img(mask, pixel4a.get_f_in_pixels())
    if False:
        cv2.imwrite("cyl_mask.jpg", cv2.normalize(cyl_mask, None, 0, 255, cv2.NORM_MINMAX))

    for i, img in tqdm(enumerate(imgs), total=len(imgs)):
        cyl_img = make_cylinder_img(img, pixel4a.get_f_in_pixels())
        cyl_img = np.clip(cyl_img, 0, 255).astype(np.int32)

        if False:
            cv2.imwrite("cyl_img_{}.jpg".format(i), cyl_img)

        if cyl_imgs:
            # There size should be the same
            assert cyl_imgs[-1].shape == cyl_img.shape
        cyl_imgs.append(cyl_img)


    print("[*] Compute the offset for each (i-1, i) pair")
    n = len(cyl_imgs)
    offsets = [None]

    for i in tqdm(range(1, n)):
        du_dv = compute_offset(cyl_imgs[i-1], cyl_mask, cyl_imgs[i], cyl_mask)
        offsets.append(du_dv)

    # Instantiate CylinderImage objects
    h, w = cyl_imgs[0].shape[0:2]
    # Assume the center of first cylinder image is at (0, 0)
    cyl_images = [CylinderImage(np.array([-(w//2), -(h//2)], dtype=np.int32), cyl_imgs[0], cyl_mask.copy())]
    for i in range(1, n):
        du_dv = offsets[i]

        prev_image = cyl_images[-1]

        # (u, v) -> (u+du, v)
        tl = prev_image.uv_top_left.copy()
        tl += du_dv

        image = CylinderImage(tl, cyl_imgs[i], cyl_mask.copy())

        cyl_images.append(image)

    # Merge all cyl_images to together
    merged = cyl_images[0]
    for i in tqdm(range(1, n)):
        merged = merged.merge(cyl_images[i])


    center_u = merged.center()[0]
    merged.write(args.out, pixel4a.get_f_in_pixels(), center_u, args.wrap)
    return

if __name__ == "__main__":
    main()
