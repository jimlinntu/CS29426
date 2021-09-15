import numpy as np
import cv2
from scipy import signal

DEBUG = True

def normalize_gradient(img):
    # gradient is in [-255, 255]
    # need to map [-255, 255] -> [0, 255]
    # by (x-(-255))/2
    img = (img - (-255))/2
    return np.clip(img, 0, 255)

def main():
    img = cv2.imread("./src_imgs/cameraman.png")
    h, w = img.shape[0:2]
    gray_img = img[:, :, 0]
    if DEBUG:
        print(f"The shape of the image: {gray_img.shape}")
        print(f"The dtype of the image: {gray_img.dtype}")

    ##### Part 1.1 ######
    Dx = np.array([[-1, 1]])
    Dy = np.array([[-1], [1]])

    if DEBUG:
        print(f"Dx: {Dx}")
        print(f"Dy: {Dy}")

    dimg_dx = cv2.filter2D(gray_img, cv2.CV_32F, Dx)
    dimg_dy = cv2.filter2D(gray_img, cv2.CV_32F, Dy)

    dimg_dx_vis = normalize_gradient(dimg_dx)
    dimg_dy_vis = normalize_gradient(dimg_dy)

    cv2.imwrite("dimg_dx.png", dimg_dx_vis)
    cv2.imwrite("dimg_dy.png", dimg_dy_vis)

    dimg_mag = np.sqrt(dimg_dx**2+dimg_dy**2)
    # normalize from [0, sqrt(2 * 255 * 255)] -> [0, 255]
    # sqrt(2 * 255 * 255) = sqrt(2) * 255
    # so divide the image maginitude by sqrt(2)
    dimg_mag_vis = dimg_mag / np.sqrt(2)
    dimg_mag_vis = np.clip(dimg_mag_vis, 0, 255)
    cv2.imwrite("dimg_mag_vis.png", dimg_mag_vis)
    _, dimg_mag_vis_bin = cv2.threshold(dimg_mag_vis.astype(np.uint8), 40, 255, cv2.THRESH_BINARY)
    cv2.imwrite("dimg_mag_vis_bin.png", dimg_mag_vis_bin)

    ##### Part 1.2 ######
    k = 3
    sigma = 0.3*((k-1)*0.5 - 1) + 0.8 # Use the setting from OpenCV doc
    if DEBUG:
        print(f"sigma: {sigma}")
    g_1d = cv2.getGaussianKernel(k, sigma)
    g_1d = g_1d.astype(np.float32)
    g_2d = g_1d.dot(g_1d.T)

    if DEBUG:
        print("Gaussian Kernel:")
        print(g_2d)

    blur = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, g_2d)
    dblur_dx = cv2.filter2D(blur, cv2.CV_32F, Dx)
    dblur_dy = cv2.filter2D(blur, cv2.CV_32F, Dy)

    dblur_dx_vis = normalize_gradient(dblur_dx)
    dblur_dy_vis = normalize_gradient(dblur_dy)

    cv2.imwrite("dblur_dx.png", dblur_dx_vis)
    cv2.imwrite("dblur_dy.png", dblur_dy_vis)


    # DoG (dx and dy)
    # NOTE: because filter2D does not support convolution in full mode, we have to use convolve2d here
    # (convolution associativity requires the two filters to be fully convolved)
    dg_2d_dx = signal.convolve2d(g_2d, cv2.flip(Dx, 1), mode="full")
    dg_2d_dy = signal.convolve2d(g_2d, cv2.flip(Dy, 1), mode="full")

    print(dg_2d_dx)
    # This will be different!!
    # vvvvvvvvvvvvvvvvvvvvvvv
    # print(signal.convolve2d(g_2d, cv2.flip(Dx, 1), mode="same"))

    dblur_dx_DoG = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, cv2.flip(dg_2d_dx, 1))

    dblur_dy_DoG = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, cv2.flip(dg_2d_dy, 1))

    print(dblur_dx[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)])
    # This will be different!!
    # vvvvvvvvvvvvvvvvvvvvvvv
    # print(cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, cv2.flip(signal.convolve2d(g_2d, cv2.flip(Dx, 1), mode="same"), 1))[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)])
    print(dblur_dx_DoG[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)])
    print("Is dblur_dx and dblur_dx_DoG similar? {}".format(np.allclose(dblur_dx[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)],
                dblur_dx_DoG[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)], atol=1.e-3)))
    print("Is dblur_dy and dblur_dy_DoG similar? {}".format(np.allclose(dblur_dy[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)],
                dblur_dy_DoG[int(0.1*h):int(0.8*h),int(0.1*w):int(0.8*w)], atol=1.e-3)))

    dblur_dx_DoG = normalize_gradient(dblur_dx_DoG)
    dblur_dy_DoG = normalize_gradient(dblur_dy_DoG)

    cv2.imwrite("dblur_dx_DoG.png", dblur_dx_DoG)
    cv2.imwrite("dblur_dy_DoG.png", dblur_dy_DoG)

if __name__ == "__main__":
    main()
