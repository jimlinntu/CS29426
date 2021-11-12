import numpy as np
import cv2

def gaussian():
    # https://math.stackexchange.com/questions/2126340/derive-the-separability-of-2d-gaussian
    x = np.arange(0, 30)
    y = np.arange(0, 30)
    xx, yy = np.meshgrid(x, y)

    mu = (max(x) + min(x)) / 2.
    sigma = 6

    # (n, n)
    dist_square = (xx - mu)**2 + (yy - mu)**2

    # (n, n) / 2 pi sigma^2
    g = np.exp(-dist_square / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    mx = np.max(g)

    # Push the largest value to 1
    g = g / mx
    return g

def paste_gaussian(keypoint, h, w):
    '''
        Paste this gaussian's center at keypoint
    '''
    assert keypoint.shape == (2,)
    prob_map = np.zeros((h, w), dtype=np.float32)

    kx, ky = keypoint.astype(np.int32)

    if kx < 0 or kx >= w or ky < 0 or ky >= h:
        return prob_map

    g = gaussian()
    gh, gw = g.shape[0:2]

    # compute the top left
    # [sx, ex], [sy, ey]
    sx, sy = kx - gw//2, ky - gh//2
    ex, ey = sx + gw - 1, sy + gh - 1

    # because this top left may be out of bound
    bound_sx, bound_sy = max(sx, 0), max(sy, 0)
    bound_ex, bound_ey = min(ex, w-1), min(ey, h-1)

    prob_map[bound_sy:bound_ey+1, bound_sx:bound_ex+1] = (
            g[bound_sy-sy:bound_ey-sy+1, bound_sx-sx:bound_ex-sx+1]
    )

    return prob_map

if __name__ == "__main__":
    g = gaussian() * 255
    g = np.clip(g, 0, 255)
    g = g.astype(np.uint8)
    h = cv2.applyColorMap(g, cv2.COLORMAP_JET)
    cv2.imwrite("gaussian.jpg", h)


    pasted = paste_gaussian(np.array([0, 0]), 100, 100)
    pasted = (pasted * 255).astype(np.uint8)
    # cv2.imwrite("paste_gaussian.jpg", pasted)
    pasted = paste_gaussian(np.array([99, 99]), 100, 100)
    pasted = (pasted * 255).astype(np.uint8)
    # cv2.imwrite("paste_gaussian.jpg", pasted)
    pasted = paste_gaussian(np.array([99, 45]), 100, 100)
    pasted = (pasted * 255).astype(np.uint8)
    cv2.imwrite("paste_gaussian.jpg", pasted)
