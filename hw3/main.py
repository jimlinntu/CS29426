import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json
import random

from skimage import draw
from scipy import interpolate


import argparse

class CPSelect():
    # https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    def __init__(self, img_left, img_right,
            pts_left=None, pts_right=None, add_corners=True,
            scale=1):
        assert isinstance(img_left, np.ndarray)
        assert isinstance(img_right, np.ndarray)

        self.img_left = img_left
        self.img_right = img_right

        h, w = img_left.shape[0:2]
        # Default adding the four corners of two images
        if pts_left is None:
            self.pts_left = [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]] if add_corners else []
        else:
            self.pts_left = pts_left

        if pts_right is None:
            self.pts_right = [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]] if add_corners else []
        else:
            self.pts_right = pts_right

        self.win_left = "img_left"
        self.win_right = "img_right"

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = scale


    def interactively_select(self):

        cv2.namedWindow(self.win_left)
        cv2.namedWindow(self.win_right)

        cv2.setMouseCallback(self.win_left, self.capture_click, {"winname": self.win_left})
        cv2.setMouseCallback(self.win_right, self.capture_click, {"winname": self.win_right})

        while True:
            self.draw(self.win_left)
            self.draw(self.win_right)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') and len(self.pts_left) == len(self.pts_right):
                break

        cv2.destroyAllWindows()


        return [self.pts_left, self.pts_right]

    def draw(self, win):
        pts = []
        if win == self.win_left:
            new_img = self.img_left.copy()
            pts = self.pts_left
        elif win == self.win_right:
            new_img = self.img_right.copy()
            pts = self.pts_right
        else:
            return

        for i, pt in enumerate(pts):
            cv2.drawMarker(new_img, pt, (0, 0, 255), markerType=cv2.MARKER_CROSS)
            cv2.putText(new_img, f"{i+1}", pt, self.font, self.scale, (0, 255, 0))

        cv2.imshow(win, new_img)

    def capture_click(self, event, x, y, flags, param):
        win = param["winname"]
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw marker
            if win == self.win_left:
                self.pts_left.append([x, y])
            else:
                self.pts_right.append([x, y])

            self.draw(win)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # undo
            if win == self.win_left:
                if self.pts_left: self.pts_left.pop()
            else:
                if self.pts_right: self.pts_right.pop()

            self.draw(win)

def randcolor():
    return tuple(random.randint(0, 255) for i in range(3))

def get_delaunay(shape_vector, h, w):
    assert isinstance(shape_vector, np.ndarray)
    assert shape_vector.shape[1] == 2 # [(x,y)...]

    # remove decimal points and convert it to float
    shape_vector = shape_vector.astype(np.int32).astype(np.float32)

    coord2idx = {}

    subdiv = cv2.Subdiv2D((0, 0, w, h))

    for idx, key_pt in enumerate(shape_vector):
        coord2idx[tuple(key_pt)] = idx
        subdiv.insert(key_pt)

    triangles = subdiv.getTriangleList()
    # Map coordinates back to the index
    tri_indices = []
    for tri in triangles:
        a, b, c = tri[0:2], tri[2:4], tri[4:6]
        a, b, c = tuple(a), tuple(b), tuple(c)

        tri_idxs = [coord2idx[a], coord2idx[b], coord2idx[c]]
        tri_indices.append(tri_idxs)

    tri_indices = np.array(tri_indices, dtype=np.int32)
    return tri_indices

def visualize_triangles(img, shape_vec, tri_indices):
    assert isinstance(tri_indices, np.ndarray)
    assert tri_indices.dtype == np.int32

    new_img = img.copy()

    for tri in tri_indices:
        a, b, c = tri

        triangle_pts = np.array([shape_vec[a], shape_vec[b], shape_vec[c]], dtype=np.int32)

        new_img = cv2.polylines(new_img, [triangle_pts], isClosed=True, color=randcolor(), thickness=2)

    return new_img

def getAffineTransform(src, dst):
    assert isinstance(src, np.ndarray)
    assert isinstance(dst, np.ndarray)
    assert src.shape == dst.shape and src.shape == (3, 2)

    src = src.astype(np.float32)
    dst = dst.astype(np.float32)

    a, b, c = src
    A, B, C = dst

    M = np.array([c-b, a-b], dtype=np.float32).T
    M_inv = np.linalg.inv(M)

    T1 = np.concatenate([M_inv, M_inv.dot(-b.reshape(2, 1))], axis=1)
    T1 = np.concatenate([T1, [[0, 0, 1]]], axis=0)

    M2 = np.array([C-B, A-B], dtype=np.float32).T

    T2 = np.concatenate([M2, B.reshape(2, 1)], axis=1)
    T2 = np.concatenate([T2, [[0, 0, 1]]], axis=0)

    T = T2.dot(T1)
    return T[0:2] # strip the last row [0, 0, 1]

def area(pts):
    assert pts.shape[1] == 2
    x, y = pts[:,0], pts[:,1]
    a = (x-np.roll(x, 1)).dot(y+np.roll(y, 1))
    a = np.abs(a)/2
    return a

def warp(img, shape_src, shape_dst, tri_indices, bg=None):
    assert isinstance(img, np.ndarray)
    assert isinstance(shape_src, np.ndarray)
    assert isinstance(shape_dst, np.ndarray)
    assert isinstance(tri_indices, np.ndarray)
    assert shape_src.shape == shape_dst.shape

    h, w = img.shape[0:2]
    Y, X = np.arange(0, h), np.arange(0, w)

    # Z = f(X, Y)
    Z = img.transpose(1, 0, 2)

    if bg is None:
        warped = np.zeros_like(img, dtype=np.float32)
    else:
        warped = bg.copy()

    for tri in tri_indices:
        a, b, c = tri

        src = np.array([shape_src[a], shape_src[b], shape_src[c]], np.float32)
        dst = np.array([shape_dst[a], shape_dst[b], shape_dst[c]], np.float32)

        if area(dst) <= 3:
            # too thin just ignore it
            continue

        # TODO: ignore it if the area is too small (<=5 ??)
        M_inv = getAffineTransform(dst, src)

        # NOTE: polygon(rows, cols) not polygon(xs, ys)!
        ys, xs = draw.polygon(dst[:, 1], dst[:, 0])

        # (n,) -> (n, 3)
        homo_coords = np.stack([xs, ys, np.ones(len(xs))], axis=1).astype(np.float32)
        assert homo_coords.shape[1] == 3

        # (n, 2)
        inverse_coords = homo_coords.dot(M_inv.T)
        assert inverse_coords.shape[1] == 2

        # Interpolate and extrapolate
        pixel_vals = interpolate.interpn((X, Y), Z, inverse_coords,
                bounds_error=False, fill_value=None)

        warped[ys, xs] = pixel_vals

    return np.clip(warped, 0, 255).astype(np.int32)

def cross_dissolve(img_left, img_right, alpha):
    assert 0. <= alpha <= 1.
    return np.clip(img_left * alpha + img_right * (1-alpha), 0, 255)

def visualize_markers(img, pts, marker=True, text=True, scale=0.3, markerSize=5):
    new_img = img.copy()
    for i, pt in enumerate(pts):
        if marker:
            cv2.drawMarker(new_img, pt, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=markerSize)
        if text:
            cv2.putText(new_img, f"{i+1}", pt, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0))

    return new_img

def main():
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=45)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--write", action="store_true", default=False)
    args = parser.parse_args()

    img_left = cv2.imread("./src_imgs/jim-crop.jpg")
    img_right = cv2.imread("./src_imgs/george_small.jpeg")

    # TODO: handle images with different sizes
    assert img_left.shape[0:2] == img_right.shape[0:2]

    corres = [[], []]

    if args.load is None:
        select = CPSelect(img_left, img_right)
        corres = select.interactively_select()
    else:
        # load
        with open(args.load, "r") as f:
            corres = json.load(f)

    assert len(corres[0]) == len(corres[1])

    if args.load is None and args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(corres, f)

    # Make it a numpy array
    corres = np.array(corres, dtype=np.float32)

    shape_left, shape_right = corres[0], corres[1]

    mean_shape = ((shape_left + shape_right) / 2).astype(np.int32)

    tri_indices = get_delaunay(mean_shape, img_left.shape[0], img_left.shape[1])

    if args.write:
        delaunay_left = visualize_triangles(img_left, shape_left, tri_indices)
        cv2.imwrite("delaunay_left.png", delaunay_left)
        delaunay_right = visualize_triangles(img_right, shape_right, tri_indices)
        cv2.imwrite("delaunay_right.png", delaunay_right)

        left_marker = visualize_markers(img_left, shape_left.astype(np.int32), marker=True, text=False, markerSize=20)
        right_marker = visualize_markers(img_right, shape_right.astype(np.int32), marker=True, text=False, markerSize=20)
        cv2.imwrite("left_marker.jpg", left_marker)
        cv2.imwrite("right_marker.jpg", right_marker)

    if False:
        # test getAffineTransform
        A = np.array([1, 2])
        B = np.array([5, 3])
        C = np.array([2, 4])
        pts1 = np.array([A, B, C], dtype=np.float32)
        pts2 = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.float32)
        pts3 = np.array([[4, 5], [6, 10], [11, 7]], dtype=np.float32)

        print(getAffineTransform(pts1, pts3))
        print(cv2.getAffineTransform(pts1, pts3))

    # Compute the midway face
    warped_left = warp(img_left, shape_left, mean_shape, tri_indices)
    warped_right = warp(img_right, shape_right, mean_shape, tri_indices)
    midway = cross_dissolve(warped_left, warped_right, 0.5)
    if args.write:
        cv2.imwrite("midway.jpg", midway)

    alphas = np.linspace(1.0, 0., 45)

    pil_frames = []
    for a in alphas:
        new_shape = a * shape_left + (1-a) * shape_right

        warped_left = warp(img_left, shape_left, new_shape, tri_indices)
        warped_right = warp(img_right, shape_right, new_shape, tri_indices)

        morphed = cross_dissolve(warped_left, warped_right, a).astype(np.uint8)

        pil_frames.append(Image.fromarray(cv2.cvtColor(morphed, cv2.COLOR_BGR2RGB)))


    if args.write:
        pil_frames[0].save(fp="output.gif", format="GIF", append_images=pil_frames[1:],
            save_all=True, duration=1/args.fps*1000, loop=0)


if __name__ == "__main__":
    main()
