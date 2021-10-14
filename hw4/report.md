# Project 4 Auto Stitching Photo Mosaics

Name: Tzu-Chuan Lin

## Part 1

### Image Rectification

1. [Image 1](https://unsplash.com/photos/z11gbBo13ro)

![](./src_imgs/ronnie.jpeg)

Rectified:

![](./demo/ronnie-rectified.jpg)


2. [Image 2](https://unsplash.com/photos/ptXFlLXuFME)

![](./src_imgs/sarah-khan.jpeg)

Rectified:

![](./demo/sarah-rectified.jpg)

### Blend the images into a mosaic

These are the pictures I have taken:

* Left:

![](./src_imgs/hall/left.jpg)

* Middle:

![](./src_imgs/hall/mid.jpg)

* Right:

![](./src_imgs/hall/right.jpg)

* Combined:

![](./demo/ihouse_merged.jpg)


### Conclusions


Q: Whats the most important/coolest thing you have learned from this part?

* I learned to derive the homography equation by hand. Previously I just directly used `cv2.findHomography`.
* I also learned how to stitch two already warped images together (by checking using their origins information).
