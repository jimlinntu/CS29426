import numpy as np
import cv2
from scipy import interpolate
from matplotlib import pyplot as plt

import argparse

def findHomography(src_pts, dst_pts):
    '''
        Solving
        A = [[x y 1 0 0 0 -xx' -x'y]
            [0 0 0 x y 1 -xy' -yy']]
        h = [h1, h2, h3, h4, h5, h6, h7, h8]
        b = [x', y']

        Solve: Ah = b
    '''

    n = len(src_pts)
    assert n == len(dst_pts)
    # (2n, 1) [x', y', x', y' ...]
    b = dst_pts.reshape(2*n)

    # (n,)
    x, y = src_pts[:, 0], src_pts[:, 1]
    x_p, y_p = dst_pts[:, 0], dst_pts[:, 1]

    x_xp = x*x_p
    xp_y = x_p*y
    x_yp = x*y_p
    y_yp = y*y_p

    # Assign x, y into the correct places
    A = np.zeros((2*n, 8), dtype=np.float64)
    A[0::2, 6] = -x_xp
    A[0::2, 7] = -xp_y
    A[1::2, 6] = -x_yp
    A[1::2, 7] = -y_yp
    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1
    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1

    # Solve it!
    h = np.linalg.lstsq(A, b, rcond=None)[0]

    H = np.array([h[0:3], h[3:6], [h[6], h[7], 1]], dtype=np.float64)
    return H

def warpPerspective(img, inv_H, dsize):
    assert isinstance(img, np.ndarray)
    assert isinstance(inv_H, np.ndarray) and inv_H.shape == (3, 3)
    assert isinstance(dsize, tuple)
    assert len(img.shape) == 3 # (h, w, c)

    # Inverse warping
    h, w, num_channels = img.shape[0:3]
    out_w, out_h = dsize

    y = np.arange(0, out_h, 1)
    x = np.arange(0, out_w, 1)

    xv, yv = np.meshgrid(x, y)

    ones = np.ones((out_h, out_w), dtype=np.float64)
    # (out_h, out_w, 3)
    homo_coords = np.stack([xv, yv, ones], axis=2)
    # get the inversed coords
    # (out_h, out_w, 3)
    coords = homo_coords.dot(inv_H.T)
    # (out_h, out_w, 3) -> (out_h, out_w, 2)
    coords = coords[:, :, :2] / coords[:, :, 2:3]

    # (out_h * out_w, 2)
    coords = coords.reshape(-1, 2)
    Z = img.transpose(1, 0, 2)
    # (out_h * out_w, num_channels)
    pixels = interpolate.interpn((np.arange(0, w), np.arange(0, h)), Z, coords,
            bounds_error=False, fill_value=np.zeros((num_channels, ), dtype=np.float32))

    out = pixels.reshape(out_h, out_w, num_channels)
    return out

def unit_test():
    h, w = 100, 100
    img = np.zeros((h, w), dtype=np.int32)
    src_pts = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
    dst_pts = np.array([[20, 20], [80, 20], [90, 90], [10, 90]], dtype=np.int32)
    H, status = cv2.findHomography(src_pts, dst_pts)
    myH = findHomography(src_pts, dst_pts)

    # Draw a rectangle
    cv2.drawContours(img, [src_pts], -1, 255, -1)
    print(H)
    print(myH)

    warped = cv2.warpPerspective(img.astype(np.float32), H.astype(np.float32), (w, h))

    invH = cv2.findHomography(dst_pts, src_pts)[0]
    my_invH = findHomography(dst_pts, src_pts)
    mywarped = warpPerspective(np.expand_dims(img, -1), my_invH, (w, h))

    cv2.imwrite("warped.jpg", warped)
    cv2.imwrite("mywarped.jpg", mywarped)


def apply_H(pts, H):
    assert isinstance(pts, np.ndarray) and len(pts.shape) == 2
    assert H.shape == (3, 3)
    # pts: (n, 2)
    # H: (3, 3)
    n = len(pts)
    ones = np.ones((n, 1))
    homo_pts = np.concatenate([pts, ones], axis=-1)
    transformed = homo_pts.dot(H.T)
    transformed[:, 0:2] /= transformed[:, 2:3]
    return transformed[:, :2]
