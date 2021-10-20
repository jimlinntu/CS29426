from homography import findHomography, apply_H
from merge import warp_toward_fixed_img, Image

import numpy as np
import cv2
import numba as nb

from skimage import feature

import argparse

def imwrite(p, img, jpg_quality):
    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

class Visualizer():
    @staticmethod
    def visualize_corners(img, corners, color):
        canvas = img.copy()
        for p in corners:
            cv2.drawMarker(canvas, p, (0, 255, 0), cv2.MARKER_STAR, 10)
        return canvas

class HarrisCorner():
    def __init__(self):
        pass

    def detect(self, img, discard_ratio=0.02):
        assert isinstance(img, np.ndarray)
        h, w = img.shape[0:2]

        xmin, xmax = int(w*discard_ratio), int(w*(1-discard_ratio))
        ymin, ymax = int(h*discard_ratio), int(h*(1-discard_ratio))

        corners = []
        corner_scores = []

        I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        response = feature.corner_harris(I, method='eps', sigma=1)
        peak_coords = feature.peak_local_max(response, min_distance=30, indices=True)
        # (r, c) -> (c, r)
        peak_coords = peak_coords[:, [1, 0]]

        mask = (xmin <= peak_coords[:, 0]) & (peak_coords[:, 0] <= xmax) \
                & (ymin <= peak_coords[:, 1]) & (peak_coords[:, 1] <= ymax)

        peak_coords = peak_coords[mask]
        cc, rr = peak_coords[:, 0], peak_coords[:, 1]

        corners.extend(peak_coords)
        corner_scores.extend(response[rr, cc])

        corners = np.array(corners)
        corner_scores = np.array(corner_scores)

        assert corners.shape[1] == 2

        assert len(corners) == len(corner_scores)
        return corners, corner_scores

@nb.jit(nopython=True)
def anms(n_desired, corners, corner_scores):
    '''
        According to http://cs.brown.edu/courses/csci1950-g/results/proj6/steveg/theory.html
        We can then sort our features based on radius size, and pull the first n features when we request a specific amount. In doing this, we aren't guaranteed to get the n features with the highest corner strengths, but instead, we get the n most dominant in their region, which ensures we get spatially distributed strong points.
    '''
    # assert isinstance(corners, np.ndarray)
    # assert isinstance(corner_scores, np.ndarray)
    # Compute each point's radius
    n = len(corner_scores)
    n_desired = min(n, n_desired)

    radius = np.full((n,), np.inf, dtype=np.float32)

    for i in range(n):
        # For each pi, find the closest pj s.t.
        # corner_scores[i] < corner_scores[j]
        for j in range(n):
            if corner_scores[i] < corner_scores[j]:
                # update the radius (to the closest larger point)
                r = np.sqrt(np.sum(np.square(corners[i] - corners[j])))
                radius[i] = min(radius[i], r)
    # Sort by the radius (from largest (inf) to the smallest)
    # and then take the first n_desired points
    # NOTE: Conceptually, assume all points have different radius
    #       if we take the first n_desired points
    #       this means we use radius[n_desired-1] as the radius
    #       so every point after [n_desired:] will be suppressed!!!
    indices = np.argsort(-radius)[:n_desired]
    return corners[indices], corner_scores[indices]

def grad(gray):
    assert len(gray.shape) == 2
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=11)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=11)
    if False:
        cv2.imwrite("Ix.jpg", cv2.normalize(Ix, None, 0, 255, cv2.NORM_MINMAX))
        cv2.imwrite("Iy.jpg", cv2.normalize(Iy, None, 0, 255, cv2.NORM_MINMAX))

    g = np.stack([Ix, Iy], axis=-1)
    return g

def get_feature_descriptors(
        img, corners, orientation=True, normalize=True):
    assert len(img.shape) == 3
    assert corners.shape[1] == 2 # (x, y)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (h, w, 2)
    dimg = grad(gray_img)

    # Descriptor's patch size and the scale
    des_h = 64
    des_s = 2
    des_H = des_h * des_s

    # (4, 2)
    box = np.array(
        [[0, 0], [des_H-1, 0],
         [des_H-1, des_H-1], [0, des_H-1]], dtype=np.float32)
    # (1, 2)
    center = np.array([[des_H//2, des_H//2]])

    descriptors = []

    for corner in corners:
        col, row = corner
        # Move this box using `corner` as the center
        b = (box + (corner.reshape(1, 2) - center)).astype(np.int32)
        if orientation:
            # (2,)
            vec = dimg[row, col]
            # Compute the angle
            theta = np.arctan2(vec[1], vec[0])
            deg = np.rad2deg(theta)
            align_b = b.astype(np.int32)
            # rotate the bounding box
            # rot_mat = cv2.getRotationMatrix2D(corner.astype(np.float32), -deg, 1.0)
            rot_mat = cv2.getRotationMatrix2D(corner.astype(np.float32), 0, 1.0)
            # oriented normalized box
            b = np.concatenate([b, np.ones((4, 1))], axis=1).dot(rot_mat.T)
            b = b.astype(np.float32)

            M = cv2.getAffineTransform(b[:3], box[:3])
            # (des_H, des_H, 3)
            des = cv2.warpAffine(img, M, (des_H, des_H), borderValue=(0, 0, 0))
            if False:
                cv2.imwrite("des-aligned.jpg",
                        img[align_b[0][0]:align_b[2][0], align_b[0][1]:align_b[2][1]])
                cv2.imwrite("des.jpg", des)
        else:
            des = np.zeros((des_H, des_H, 3), dtype=np.float32)
            top_left, bot_right = b[0], b[2]

            if False:
                # debug
                if any(top_left < 0):
                    import pdb; pdb.set_trace()
                if any(bot_right >= np.array([img.shape[1], img.shape[0]])):
                    import pdb; pdb.set_trace()

            # when top_left < [0, 0], we need to change the 
            # starting corner for `des`
            offset = np.maximum(top_left, [0, 0]) - top_left

            tl = np.maximum(top_left, [0, 0])
            br = np.minimum(bot_right, [img.shape[1]-1, img.shape[0]-1])
            # effective h and w
            trunc_w, trunc_h = (br - tl +1)
            des[offset[1]:offset[1]+trunc_h, offset[0]:offset[0]+trunc_w, :] =\
                img[tl[1]:tl[1]+trunc_h, tl[0]:tl[0]+trunc_w, :]

        des = des.astype(np.float32)
        # Resize
        des = cv2.resize(des, (des_h, des_h))

        # Normalize mean and variance (over each channel)
        if normalize:
            mean = np.mean(des.reshape(des_h*des_h, 3), axis=0).reshape(1, 1, 3)
            std = np.std(des.reshape(des_h*des_h, 3), axis=0).reshape(1, 1, 3)
            des = (des - mean) / (std + 1e-5)
        descriptors.append(des)

    # (n corners, des_h, des_h, 3)
    descriptors = np.stack(descriptors, axis=0)
    return descriptors

@nb.jit(nopython=True)
def pair_corners(des1, des2):
    n = len(des1)
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # first nearest neighbor distance / second nearest neighbor
    threshold_ratio = 0.8

    candidates = set()
    for j in range(n):
        candidates.add(j)

    matches = []
    # Match 
    for i in range(n):
        # Corner i
        best = -1
        best_score = np.inf
        second = -1
        second_best_score = np.inf

        for j in range(n):
            # Corner j

            # already selected
            if j not in candidates:
                continue
            # Compute the ssd score
            score = np.sqrt(np.sum(np.square(des1[i]-des2[j])))

            if score < best_score:
                second = best
                second_best_score = best_score

                best = j
                best_score = score
            elif score < second_best_score:
                # record the second best
                second = j
                second_best_score = score

        # if False:
        #     cv2.imwrite("i_des.jpg", des1[i])
        #     cv2.imwrite("best_des.jpg", des2[best])
        #     cv2.imwrite("second_des.jpg", des2[second])
        #     print(best_score / second_best_score)
        # if best_score / second_best_score < threshold => Good match!
        if best_score < second_best_score * threshold_ratio:
            # record a match
            matches.append([i, best])
            # delete j from the candidate
            candidates.remove(best)

    # (# of good pairs, 2 (i, j))
    return np.array(matches, dtype=np.int32)

def ransac_mask(matches, cor1, cor2):
    # matches: (n, 2)

    # convert (i, j) match to points
    n = len(matches)
    pool = np.arange(n)
    pts1 = []
    pts2 = []
    for match in matches:
        i, j = match
        pts1.append(cor1[i])
        pts2.append(cor2[j])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    error = 10

    best_num_inliers = -1
    best_inlier_mask = np.zeros((n, ), dtype=np.bool)

    for i in range(1000):
        # Randomly choose 4 pairs to compute the homography
        indices = np.random.choice(pool, 4, replace=False)
        src_pts = pts1[indices]
        dst_pts = pts2[indices]

        H = findHomography(src_pts, dst_pts)
        # Compute inliers
        new_pts2 = apply_H(pts1, H)

        # Agreement mask
        inlier_mask = np.sqrt(np.sum((pts2 - new_pts2) **2, axis=-1)) < error
        num_inliers = np.sum(inlier_mask)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask

    # Compute the homography on the all inliers
    return best_inlier_mask

def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed", type=str, help="The fixed image (usually the middle)")
    parser.add_argument("out", type=str)
    parser.add_argument("--imgs", nargs="+", required=True)
    parser.add_argument("--anms", type=int, default=1000)
    args = parser.parse_args()

    fixed = cv2.imread(args.fixed)
    imgs = []
    for p in args.imgs:
        I = cv2.imread(p)
        imgs.append(I)

    hc = HarrisCorner()
    corners, scores = hc.detect(fixed)
    before = corners
    corners, _ = anms(args.anms, corners, scores)
    des = get_feature_descriptors(fixed, corners, orientation=False)

    if False:
        canvas = fixed.copy()
        for p in before:
            cv2.drawMarker(canvas, p, (0, 255, 0), cv2.MARKER_STAR, 10)
        for p in corners:
            cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_STAR, 10)

        d = get_feature_descriptors(fixed, corners, orientation=False, normalize=False)

        imwrite("corners.jpg", canvas, 20)
        max_col = 50
        n = d.shape[0]
        w = d.shape[1]

        # (n, w, w, 3) -> (n // max_col, max_col, w, w, 3)
        d = d.reshape(n // max_col, max_col, w, w, 3)
        # (n//max_col, w, max_col, w, 3) -> (n // max_col * w, max_col * w, 3)
        d = d.transpose(0, 2, 1, 3, 4)
        d = d.reshape(-1, max_col * w, 3)
        imwrite("des.jpg", d, 20)

    warped_images = []
    for idx, img in enumerate(imgs):
        print("Processing image {}....".format(idx))
        print("Stiching image...")
        cur_corners, scores = hc.detect(img)
        cur_corners, _ = anms(args.anms, cur_corners, scores)
        cur_des = get_feature_descriptors(img, cur_corners, orientation=False)

        print("Pairing the corners...")
        matches = pair_corners(des, cur_des)
        inlier_mask = ransac_mask(matches, corners, cur_corners)
        matches = matches[inlier_mask]
        print("Successfully matches {} pairs".format(len(matches)))

        # Map moved_pts to fixed_pts
        fixed_pts = np.array([corners[i] for (i, j) in matches])
        moved_pts = np.array([cur_corners[j] for (i, j) in matches])

        print("Warping the image...")
        warped_image =\
            warp_toward_fixed_img(fixed, img, fixed_pts, moved_pts)
        warped_images.append(warped_image)

    # Warped all images
    merged = Image(fixed,
            np.ones((*fixed.shape[0:2], 1), dtype=np.int32),
            np.zeros((2, ), dtype=np.int32))
    for image in warped_images:
        merged = merged.merge(image)

    merged.write(args.out, 20)
    return

if __name__ == "__main__":
    main()
