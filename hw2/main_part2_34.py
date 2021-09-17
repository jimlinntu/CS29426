import numpy as np
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from pathlib import Path

DEBUG = False

def get2d_gaussian(ksize, sigma):
    g1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = g1d.dot(g1d.T)
    return kernel

def get_gaussian_stacks(img, ksize, depth):
    stacks = [img]

    prev_img = img.astype(np.float32)
    # https://computergraphics.stackexchange.com/questions/256/is-doing-multiple-gaussian-blurs-the-same-as-doing-one-larger-blur
    for i in range(depth):
        kernel = get2d_gaussian(ksize * (i+1), 0)
        # kernel = get2d_gaussian(ksize, 0)
        cur = cv2.filter2D(prev_img, cv2.CV_32F, kernel)

        stacks.append(cur)
        prev_img = cur
    return stacks

def normalize(img):
    # NOTE: have to normalize each channel separately!!
    b = cv2.normalize(img[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.normalize(img[:, :, 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r = cv2.normalize(img[:, :, 2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)

def get_laplacian_stacks(img, ksize, depth):
    lp_stacks = []
    prev_img = img.astype(np.float32)

    for i in range(depth):
        kernel = get2d_gaussian(ksize * (i+1), 0)
        # kernel = get2d_gaussian(ksize , 0)
        cur = cv2.filter2D(prev_img, cv2.CV_32F, kernel)
        lp = prev_img - cur

        prev_img = cur

        lp_stacks.append(lp)

    lp_stacks.append(prev_img)

    return lp_stacks

def bgr2rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=str)
    parser.add_argument("right", type=str)
    parser.add_argument("depth", type=int)
    parser.add_argument("mask_ksize", type=int)
    parser.add_argument("left_ksize", type=int)
    parser.add_argument("right_ksize", type=int)
    parser.add_argument("out_dir", type=str)

    parser.add_argument("--mask_path", type=str, default=None, help="User provided mask")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    left_path_stem = Path(args.left).stem
    right_path_stem = Path(args.right).stem

    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    assert left.shape == right.shape

    h, w = left.shape[0:2]
    if args.mask_path is None:
        mask = np.zeros((h, w, 3), dtype=np.float32)

        # A step function mask
        mask[:, :w//2] = 1
    else:
        assert isinstance(args.mask_path, str)
        # user provided mask
        mask = cv2.imread(args.mask_path)
        mask = mask / 255 # 
        mask = mask.astype(np.float32)

    assert mask.shape[0:2] == (h, w)

    # depth = 4
    depth = args.depth

    # mask_stacks = get_gaussian_stacks(mask, 71, depth)
    mask_stacks = get_gaussian_stacks(mask, args.mask_ksize, depth)

    left_g_stacks = get_gaussian_stacks(left, args.left_ksize, depth)
    right_g_stacks = get_gaussian_stacks(right, args.right_ksize, depth)

    left_lp_stacks = get_laplacian_stacks(left, args.left_ksize, depth)
    right_lp_stacks = get_laplacian_stacks(right, args.right_ksize, depth)

    for i, m in enumerate(mask_stacks):
        cv2.imwrite(str(out_dir / f"mask_{i}.jpg"), (m * 255).astype(np.uint8))

    for i, l in enumerate(left_g_stacks):
        cv2.imwrite(str(out_dir / f"{left_path_stem}_{i}.jpg"), l)

    for i, r in enumerate(right_g_stacks):
        cv2.imwrite(str(out_dir / f"{right_path_stem}_{i}.jpg"), r)

    for i, l in enumerate(left_lp_stacks):
        cv2.imwrite(str(out_dir / f"{left_path_stem}_lp_{i}.jpg"), normalize(l))

    for i, r in enumerate(right_lp_stacks):
        cv2.imwrite(str(out_dir / f"{right_path_stem}_lp_{i}.jpg"), normalize(r))

    # merge
    col = 3

    nrow, ncol = depth+2, col

    # https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    # gs = matplotlib.gridspec.GridSpec(nrow, ncol,
    #      wspace=0.0, hspace=0.5,
    #      top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
    #      left=0.5/(ncol+1), right=1-0.5/(ncol+1))

    gs = matplotlib.gridspec.GridSpec(nrow, ncol)

    fig = plt.figure(figsize=(15, 15), dpi=100)

    out = np.zeros((h, w, 3), dtype=np.float32)

    left_acc = np.zeros((h, w, 3), dtype=np.float32)
    right_acc = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(depth, -1, -1):
        left_lp_masked = mask_stacks[i] * (left_lp_stacks[i])
        right_lp_masked = (1-mask_stacks[i]) * (right_lp_stacks[i])
        t = left_lp_masked + right_lp_masked

        left_acc += left_lp_masked
        right_acc += right_lp_masked

        ax_left = plt.subplot(gs[depth-i, 0])
        ax_right = plt.subplot(gs[depth-i, 1])
        ax_merged = plt.subplot(gs[depth-i, 2])

        ax_left.axis("off")
        ax_right.axis("off")
        ax_merged.axis("off")

        ax_left.title.set_text("Level: {}".format(i))
        ax_right.title.set_text("Level: {}".format(i))
        ax_merged.title.set_text("Level: {}".format(i))

        out += t

        ax_left.imshow(bgr2rgb(normalize(left_lp_masked)))
        ax_right.imshow(bgr2rgb(normalize(right_lp_masked)))
        ax_merged.imshow(bgr2rgb(normalize(t)))

    ax_left = plt.subplot(gs[depth+1, 0])
    ax_right = plt.subplot(gs[depth+1, 1])
    ax_merged = plt.subplot(gs[depth+1, 2])

    ax_left.axis("off")
    ax_right.axis("off")
    ax_merged.axis("off")

    ax_left.imshow(bgr2rgb(normalize(left_acc)))
    ax_right.imshow(bgr2rgb(normalize(right_acc)))
    ax_merged.imshow(bgr2rgb(normalize(out)))

    ax_left.title.set_text("Left accumulated result")
    ax_right.title.set_text("Right accumulated result")
    ax_merged.title.set_text("Final result")

    fig.savefig(str(out_dir / f"{left_path_stem}_{right_path_stem}_process.jpg"))

    if DEBUG:
        print("out: max: {}, min: {}".format(np.max(out), np.min(out)))
    out = np.clip(out, 0, 255)
    cv2.imwrite(str(out_dir / f"{left_path_stem}_{right_path_stem}_out.jpg"), out)


if __name__ == "__main__":
    main()
