# Project 4 Auto Stitching Photo Mosaics

* Name: Tzu-Chuan Lin
* SID: 3036360742
* Website: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj4/cs194-26-abu/>
* Email: tzu-chuan_lin@berkeley.edu

## Part 1

### Image Rectification

`python main.py ./src_imgs/ronnie.jpeg ronnie-rectified.jpg --load ronnie.json`
`python main.py ./src_imgs/sarah-khan.jpeg sarah-rectified.jpg --load sarah.json`

### Blend the images into a mosaic

`python main_mosaic.py src_imgs/hall/left.jpg src_imgs/hall/mid.jpg src_imgs/hall/right.jpg ihouse_merged.jpg src_imgs/hall/left_pts.mat src_imgs/hall/mid_pts.mat src_imgs/hall/mid_mid_right_pts.mat  src_imgs/hall/right_mid_right_pts.mat`
