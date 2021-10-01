import numpy as np
import cv2

from pathlib import Path
import re
import argparse
import json
import random

from main import CPSelect, get_delaunay, warp, visualize_markers

def load_pts(path):
    assert isinstance(path, Path)
    pts = []
    with path.open("r") as f:
        prev_idx = -1
        for line in f:
            if "#" in line: continue
            line = line.strip()
            if len(line) == 0: continue
            split = line.split()
            if len(split) < 7:
                continue

            x, y, idx = float(split[2]), float(split[3]), float(split[4])

            pts.append([x, y])

    return np.array(pts, dtype=np.float32)

def crop(img, pts, target_h=350, target_w=240):
    assert isinstance(img, np.ndarray)
    assert isinstance(pts, np.ndarray)

    h, w = img.shape[0:2]

    half_h, half_w = target_h // 2, target_w // 2

    # (2, )
    center = np.mean(pts, axis=0).astype(np.int32)

    top_left = center - np.array([half_w-1, half_h-1])
    bot_right = center + np.array([half_w+1, half_h+1])

    # fix it if they are out of bound
    # fix x
    if top_left[0] < 0:
        dx = -top_left[0]
        top_left[0] = 0
        bot_right[0] += dx

    if bot_right[0] > w:
        dx = bot_right[0] - w
        top_left[0] -= dx
        bot_right[0] = w

    # fix y
    if top_left[1] < 0:
        dy = -top_left[1]
        top_left[1] = 0
        bot_right[1] += dy

    if bot_right[1] > h:
        dy = bot_right[1]-h
        top_left[1] -= dy
        bot_right[1] = h

    top_left = np.clip(top_left, [0, 0], [w, h])
    bot_right = np.clip(bot_right, [0, 0], [w, h])

    cropped = img[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]

    assert cropped.shape[0:2] == (target_h, target_w), "Need to set target_h, target_w to a lower size"

    pts = pts - top_left # new coordinates

    return cropped, pts


def interpolate(x1, x2, alpha):
    assert isinstance(x1, np.ndarray)
    assert isinstance(x2, np.ndarray)
    return alpha * x1 + (1-alpha) * x2

def main():
    random.seed(115813)
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--write", action="store_true", default=False)
    args = parser.parse_args()

    d = Path("./imm_face")
    # These images are of gray scale, remove them
    blacklist = ["02", "03", "04"]

    pattern = re.compile("[0-9]{2}-[1]m")

    names = []
    imgs = []
    shapes = []
    for p in d.iterdir():
        if p.suffix != ".jpg": continue
        if pattern.search(p.name) is None: continue

        isblack = False
        for black in blacklist:
            if black in p.name:
                isblack = True
                break
        if isblack:
            continue

        img_path = p
        pts_path = p.with_suffix(".asf")
        img = cv2.imread(str(img_path))
        pts = load_pts(pts_path)
        h, w = img.shape[0:2]
        pts = pts * np.array([[w, h]], dtype=np.float32)
        pts = pts.astype(np.int32)

        cropped, crop_pts = crop(img, pts)

        # cv2.imwrite("./{}.jpg".format(p.with_suffix("").name + "-cropped"), cropped)
        # cv2.imwrite("./{}_marker.jpg".format(p.with_suffix("").name), visualize_markers(cropped, crop_pts))

        names.append(p.with_suffix("").name)
        imgs.append(cropped)
        shapes.append(crop_pts)

    h, w = imgs[0].shape[0:2]
    # (n, h, w, 3)
    imgs = np.array(imgs, dtype=np.int32)
    # (n, 58, 2)
    shapes = np.array(shapes, dtype=np.int32)

    my_img = cv2.imread(args.img_path)

    if args.load is None:
        select = CPSelect(my_img, imgs[0].astype(np.uint8), pts_left=None, pts_right=shapes[0],
                add_corners=False, scale=0.3)
        my_shape, _ = select.interactively_select()

        if args.save is not None:
            with open(args.save, "w") as f:
                json.dump(my_shape, f)
    else:
        with open(args.load, "r") as f:
            my_shape = json.load(f)

    my_shape = np.array(my_shape, np.int32)

    # Compute the average shape

    mean_shape = np.mean(shapes, axis=0).astype(np.int32)
    vis_mean_text = visualize_markers(np.zeros_like(imgs[0]), mean_shape, marker=False, text=True)
    vis_mean_marker = visualize_markers(np.zeros_like(imgs[0]), mean_shape, marker=True, text=False)
    if args.write:
        cv2.imwrite("vis_mean_text.png", vis_mean_text)
        cv2.imwrite("vis_mean_marker.png", vis_mean_marker)

    tri_indices = get_delaunay(mean_shape, imgs[0].shape[0], imgs[0].shape[1])

    # Morph every face into that shape and then take the mean

    warpeds = []
    for img, shape in zip(imgs, shapes):
        warped = warp(img, shape, mean_shape, tri_indices)

        warpeds.append(warped)

    warpeds = np.array(warpeds)
    mean_img = np.mean(warpeds, axis=0)
    mean_img_w_marker = visualize_markers(mean_img, mean_shape, marker=True, text=False)
    if args.write:
        cv2.imwrite("mean_img.jpg", mean_img)
        cv2.imwrite("mean_img_w_marker.jpg", mean_img_w_marker)

    # sample some of them
    sample_indices = random.sample(range(0, len(names)), 5)
    for i in sample_indices:
        name = names[i]

        if args.write:
            cv2.imwrite("{}-cropped.jpg".format(name), imgs[i])
            cv2.imwrite("{}-warped.jpg".format(name), warpeds[i])

    # Visualize my key points
    my_img_w_markers = visualize_markers(my_img, my_shape, text=False)
    if args.write:
        cv2.imwrite("my_img_w_markers.jpg", my_img_w_markers)
    # Warp my shape into the mean shape
    warped_my_img = warp(my_img, my_shape, mean_shape, tri_indices)
    if args.write:
        cv2.imwrite("warped_my_img.jpg", warped_my_img)
    # Warp the mean image into the my shape
    mean_img_my_shape = warp(mean_img, mean_shape, my_shape, tri_indices)
    if args.write:
        cv2.imwrite("mean_img_my_shape.jpg", mean_img_my_shape)

    # Exaggerate between my face and the dane face
    alphas = [2.0, 1.5, 1, 0.5]
    cars = []
    for alpha in alphas:
        new_shape = interpolate(my_shape, mean_shape, alpha)
        # avoid negative coordinates
        new_shape = np.clip(new_shape, [[0, 0]], [[w, h]])

        warped_my = warp(my_img, my_shape, new_shape, tri_indices)
        warped_mean = warp(mean_img, mean_shape, new_shape, tri_indices)

        c = interpolate(warped_my, warped_mean, alpha)
        c = np.clip(c, 0, 255).astype(np.uint8)
        cars.append(c)

    for car, alpha in zip(cars, alphas):
        if args.write:
            cv2.imwrite("car-{}.jpg".format(alpha), car)

if __name__ == "__main__":
    main()
