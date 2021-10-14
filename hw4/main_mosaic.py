import numpy as np
import cv2
from scipy import io

from main import findHomography, warpPerspective, apply_H

import argparse

DEBUG = False

class Image:
    def __init__(self, img, mask, origin):
        '''
          origin (might not be (0, 0))
            -------------------->
            |
            |       img, mask
            |
            |
            v
        '''
        assert len(img.shape) == 3
        assert len(mask.shape) == 3 and mask.shape[2] == 1
        assert origin.shape == (2, ) and origin.dtype == np.int32

        self.img = img.astype(np.int32)
        self.mask = mask
        self.origin = origin
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.bot_right = origin + np.array([self.w-1, self.h-1], dtype=np.int32)

    def merge(self, image2):
        assert isinstance(image2, Image)
        # determine the new origin, the merged size

        merged_origin = np.minimum(self.origin, image2.origin)
        merged_bot_right = np.maximum(self.bot_right, image2.bot_right)

        # NOTE: +1 is important because boundaries are inclusive
        merged_w, merged_h = (merged_bot_right - merged_origin + 1).astype(np.int32)

        # Merge two images according to their mask
        canvas = np.zeros((merged_h, merged_w, 3), dtype=np.float32)
        mask = np.zeros((merged_h, merged_w, 3), dtype=np.float32)

        canvas2 = np.zeros((merged_h, merged_w, 3), dtype=np.float32)
        mask2 = np.zeros((merged_h, merged_w, 3), dtype=np.float32)

        out = np.zeros((merged_h, merged_w, 3), dtype=np.float32)

        # Paste the first image under the new coordinate system
        x, y = self.origin - merged_origin
        canvas[y:y+self.h, x:x+self.w, :] = self.img
        # mask broadcasting
        mask[y:y+self.h, x:x+self.w, :] = self.mask

        x2, y2 = image2.origin - merged_origin
        canvas2[y2:y2+image2.h, x2:x2+image2.w, :] = image2.img
        # mask broadcasting
        mask2[y2:y2+image2.h, x2:x2+image2.w, :] = image2.mask

        # Weighted average between two images intersection
        select = (mask > 0) & (mask2 > 0)
        out[select] = \
            (mask[select]*canvas[select] + mask2[select] * canvas2[select]) / (mask[select] + mask2[select])

        # Only one part exists
        select = ((mask > 0) ^ ((mask > 0) & (mask2 > 0))).astype(np.bool)
        out[select] = canvas[select]

        select = ((mask2 > 0) ^ ((mask > 0) & (mask2 > 0))).astype(np.bool)
        out[select] = canvas2[select]

        out = np.clip(out, 0, 255).astype(np.int32)
        return Image(out, (mask+mask2)[:, :, 0:1], merged_origin)

    def write(self, path, jpg_quality):
        cv2.imwrite(path, self.img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

def get_dist_mask(mask):
    assert isinstance(mask, np.ndarray)
    assert len(mask.shape) == 3
    h, w = mask.shape[0:2]
    hh, ww = h+2, w+2
    # put 0 on the borders
    padded_mask = np.zeros((hh, ww, 1), dtype=np.uint8)
    padded_mask[1:-1, 1:-1, :] = mask
    dst = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 3)
    return dst[1:-1, 1:-1, np.newaxis]

def get_corners(img):
    h, w = img.shape[0:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)
    return corners

def get_hw_from_corners(corners):
    corners = corners.astype(np.int32)
    w, h = (np.max(corners, axis=0) - np.min(corners, axis=0) + 1)
    return w, h

def warp_toward_fixed_img(fixed_img, move_img, fixed_pts, move_pts):
    assert len(fixed_img.shape) == 3
    assert len(move_img.shape) == 3
    assert len(fixed_pts) == len(move_pts)
    H = findHomography(move_pts, fixed_pts)

    corners = get_corners(move_img)
    # NOTE: we use the coordinate system of fixed_img
    #       so this new corners might be out of bound (Ex. <0)
    new_corners = apply_H(corners, H)

    # (1, 2)
    new_origin = new_corners.min(axis=0, keepdims=True).astype(np.int32)

    # map our move_pts to the fixed_pts under the new coordinate system
    inv_H = findHomography(fixed_pts - new_origin, move_pts)

    new_w, new_h = get_hw_from_corners(new_corners)
    mask = np.ones((move_img.shape[0], move_img.shape[1], 1), dtype=np.int32)
    mask = get_dist_mask(mask.astype(np.uint8)).astype(np.int32)

    warped = warpPerspective(move_img, inv_H, (new_w, new_h))
    warped = np.clip(warped.astype(np.int32), 0, 255)

    warped_mask = warpPerspective(mask, inv_H, (new_w, new_h))

    if False:
        cv2.imwrite("warped.jpg", warped)
        normalized_mask = cv2.normalize(warped_mask, None, 0, 255, cv2.NORM_MINMAX) 
        cv2.imwrite("warped_mask.jpg", normalized_mask)
    return Image(warped, warped_mask, new_origin.reshape(2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=str)
    parser.add_argument("mid", type=str)
    parser.add_argument("right", type=str)
    parser.add_argument("out", type=str)

    parser.add_argument("left_pts", type=str)
    parser.add_argument("mid_pts", type=str)

    parser.add_argument("mid_mid_right_pts", type=str)
    parser.add_argument("right_mid_right_pts", type=str)

    args = parser.parse_args()

    # Load images
    left = cv2.imread(args.left)
    mid = cv2.imread(args.mid)
    right = cv2.imread(args.right)

    # Load left <-> mid correspondences
    left_pts = io.loadmat(args.left_pts)["left_pts"].astype(np.int32)
    mid_pts = io.loadmat(args.mid_pts)["mid_pts"].astype(np.int32)

    # Load mid <-> right correspondences
    mid_mid_right_pts = io.loadmat(args.mid_mid_right_pts)["mid_mid_right_pts"].astype(np.int32)
    right_mid_right_pts = io.loadmat(args.right_mid_right_pts)["right_mid_right_pts"].astype(np.int32)

    left_image = warp_toward_fixed_img(mid, left, mid_pts, left_pts)

    right_image = warp_toward_fixed_img(mid, right, mid_mid_right_pts, right_mid_right_pts)

    mid_image = Image(mid, np.ones((*mid.shape[0:2], 1), dtype=np.int32), np.zeros((2, ), dtype=np.int32))

    # Merge mid and left
    merged_image = mid_image.merge(left_image)
    # Merge merged and right
    merged_image = merged_image.merge(right_image)

    merged_image.write(args.out, 60)

    return

if __name__ == "__main__":
    main()

