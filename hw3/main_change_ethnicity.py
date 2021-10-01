import numpy as np
import cv2

import json
import argparse
import random

from main import CPSelect, get_delaunay, visualize_markers, visualize_triangles, cross_dissolve, warp

def morph(
        img, face_mask, eth_left, eth_right,
        shape, eth_left_shape, eth_right_shape, tri_indices,
        shape_alpha, appear_alpha, h, w):

    eth_dshape = eth_right_shape - eth_left_shape

    center = shape.mean(axis=0, keepdims=True)
    out_shape = shape + shape_alpha * eth_dshape
    out_shape = np.clip(out_shape, [[0, 0]], [[w, h]])

    new_center = out_shape.mean(axis=0, keepdims=True)

    warped_eth_left = warp(eth_left, eth_left_shape, out_shape, tri_indices)
    warped_eth_right = warp(eth_right, eth_right_shape, out_shape, tri_indices)

    d_appear = warped_eth_right - warped_eth_left

    # Warp the mask so that we can know where is the face located
    mask = warp(face_mask.reshape(h, w, 1), shape, out_shape, tri_indices)

    out_img = warp(img, shape, out_shape, tri_indices, bg=img if shape_alpha == 0 else None)

    # Only change the face region!
    out_img = out_img + appear_alpha * (d_appear * mask)
    out_img = np.clip(out_img, 0, 255)

    return out_img

def get_face_mask(face_shape, h, w, tri_indices):
    assert isinstance(face_shape, np.ndarray)
    assert face_shape.shape[1] == 2

    face_shape = face_shape.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.int32)

    contours = []
    n = len(face_shape)
    for tri in tri_indices:
        a, b, c = tri

        if any([a >= n, b >= n, c >= n]):
            continue
        a, b, c = face_shape[a], face_shape[b], face_shape[c]

        contours.append([a, b, c])

    contours = np.array(contours)
    # Fill each triangle
    cv2.drawContours(mask, contours, -1, 1, -1)
    return mask

def add_corners(shape, h, w):
    assert isinstance(shape, np.ndarray)
    assert shape.shape[1] == 2

    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1], [w//2, 0], [w-1, h//2], [w//2, h-1], [0, h//2]], dtype=np.int32)
    return np.concatenate([shape, corners], axis=0)

def main():
    random.seed(115813)
    parser = argparse.ArgumentParser()
    parser.add_argument("shape_alpha", type=float)
    parser.add_argument("appear_alpha", type=float)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--write", action="store_true", default=False)
    parser.add_argument("--add_corners", action="store_true", default=False)
    args = parser.parse_args()

    img  = cv2.imread("./src_imgs/jim-crop-bell.jpeg")
    eth_left = cv2.imread("./src_imgs/averagetaiwanesemale.jpeg")
    eth_right = cv2.imread("./src_imgs/whiteamericanmale-adjust.jpeg")

    assert img.shape[0:2] == eth_left.shape[0:2] == eth_right.shape[0:2]

    if args.load is None:
        select = CPSelect(eth_left, eth_right)
        eth_left_shape, eth_right_shape = select.interactively_select()

        select2 = CPSelect(img, eth_left, pts_right=eth_left_shape)
        shape, _ = select2.interactively_select()

        shapes = {"src": shape, "eth_left": eth_left_shape, "eth_right": eth_right_shape}
    else:
        with open(args.load, "r") as f:
            shapes = json.load(f)
    if args.load is None and args.save is not None:
        with open(args.save, "w") as f:
            json.dump(shapes, f)

    h, w = img.shape[0:2]
    # Visualize the triangulation
    shape = np.array(shapes["src"], dtype=np.int32)
    face_shape = shape.copy()
    eth_left_shape = np.array(shapes["eth_left"], dtype=np.int32)
    eth_right_shape = np.array(shapes["eth_right"], dtype=np.int32)

    if args.add_corners:
        shape = add_corners(shape, h, w)
        eth_left_shape = add_corners(eth_left_shape, h, w)
        eth_right_shape = add_corners(eth_right_shape, h, w)

    img_markers = visualize_markers(img, shape, marker=True, text=False, scale=0.4, markerSize=20)
    if args.write:
        cv2.imwrite("markers.jpg", img_markers)

    tri_indices = get_delaunay(shape, h+1, w+1)
    img_tri = visualize_triangles(img, shape, tri_indices)
    if args.write:
        cv2.imwrite("img_tri.jpg", img_tri)

    out = morph(img, get_face_mask(face_shape, h, w, tri_indices), eth_left, eth_right,
            shape, eth_left_shape, eth_right_shape, tri_indices,
            args.shape_alpha, args.appear_alpha, h, w)

    cv2.imwrite("out-shape-{}-appear-{}.jpg".format(args.shape_alpha, args.appear_alpha),
            out)


if __name__ == "__main__":
    main()
