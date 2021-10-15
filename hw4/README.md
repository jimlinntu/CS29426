# Project 4 Auto Stitching Photo Mosaics

* Name: Tzu-Chuan Lin
* SID: 3036360742
* Website: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj4A/cs194-26-abu/>
* Email: tzu-chuan_lin@berkeley.edu

## Part 1

### Image Rectification

`python main.py ./src_imgs/ronnie.jpeg ronnie-rectified.jpg --load ronnie.json`
`python main.py ./src_imgs/sarah-khan.jpeg sarah-rectified.jpg --load sarah.json`

### Blend the images into a mosaic

1. `python main_mosaic.py src_imgs/hall/left.jpg src_imgs/hall/mid.jpg src_imgs/hall/right.jpg ihouse_merged.jpg src_imgs/hall/left_pts.mat src_imgs/hall/mid_pts.mat src_imgs/hall/mid_mid_right_pts.mat  src_imgs/hall/right_mid_right_pts.mat`

2. `python main_mosaic.py src_imgs/library/left.jpg src_imgs/library/mid.jpg src_imgs/library/right.jpg lib_merged.jpg src_imgs/library/left_pts.mat src_imgs/library/mid_pts.mat src_imgs/library/mid_mid_right_pts.mat  src_imgs/library/right_mid_right_pts.mat`

3. `python main_mosaic.py src_imgs/plaza/left.jpg src_imgs/plaza/mid.jpg src_imgs/plaza/right.jpg plaza_merged.jpg src_imgs/plaza/left_pts.mat src_imgs/plaza/mid_pts.mat src_imgs/plaza/mid_mid_right_pts.mat  src_imgs/plaza/right_mid_right_pts.mat`
