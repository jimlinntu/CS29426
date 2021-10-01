import numpy as np
import cv2

import json
import argparse
import random

from main import CPSelect, get_delaunay, visualize_markers, visualize_triangles, cross_dissolve, warp

def morph(
        img, eth_left, eth_right,
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

    out_img = warp(img, shape, out_shape, tri_indices, bg=img if shape_alpha == 0 else None)

    out_img = out_img + appear_alpha * d_appear
    out_img = np.clip(out_img, 0, 255)

    return out_img

def main():
    random.seed(115813)
    parser = argparse.ArgumentParser()
    parser.add_argument("shape_alpha", type=float)
    parser.add_argument("appear_alpha", type=float)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--write", action="store_true", default=False)
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
        # TODO: load
        with open(args.load, "r") as f:
            shapes = json.load(f)
    if args.load is None and args.save is not None:
        with open(args.save, "w") as f:
            json.dump(shapes, f)

    h, w = img.shape[0:2]
    # Visualize the triangulation
    shape = np.array(shapes["src"], dtype=np.int32)
    eth_left_shape = np.array(shapes["eth_left"], dtype=np.int32)
    eth_right_shape = np.array(shapes["eth_right"], dtype=np.int32)

    img_markers = visualize_markers(img, shape, marker=True, text=False, scale=0.4, markerSize=20)
    if args.write:
        cv2.imwrite("markers.jpg", img_markers)

    tri_indices = get_delaunay(shape, h+1, w+1)
    img_tri = visualize_triangles(img, shape, tri_indices)
    if args.write:
        cv2.imwrite("img_tri.jpg", img_tri)

    out = morph(img, eth_left, eth_right,
            shape, eth_left_shape, eth_right_shape, tri_indices,
            args.shape_alpha, args.appear_alpha, h, w)

    cv2.imwrite("out-shape-{}-appear-{}.jpg".format(args.shape_alpha, args.appear_alpha),
            out)


if __name__ == "__main__":
    main()
