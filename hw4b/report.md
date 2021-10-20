# Proj4B: Feature Matching for Autostitching

## Detecting corner features in an image

In my implementation, I just directly tranformed an image into gray scale and then performed the edge detection.

The result of harris corner detection + ANMS(Adaptive Non-Maximal Suppression)

NOTE: **Red** points are the points still there after ANMS.

![](./demo/corners.jpg)

## Extracting a Feature Descriptor for each feature point

The feature descriptors (before normalization):

NOTE: Because my images are with high-resolution, I set the patch size be `64x64` (resized from `128x128` patch) to increase the descriptiveness of each patch.

![](./demo/des.jpg)

## Matching these feature descriptors between two images

I used SSD (i.e.`np.sum((img1-img2)**2)`) to measure the similarity between feature two descriptors.

## Use a robust method (RANSAC) to compute a homography 

In my implementation, I gave RANSAC 1000 iterations.

## Mosaic

1. I-House's great hall:

|PartA (manual labeling)|PartB (automatic pairing)|
|---|---|
|<img src="./demo/ihouse_merged.jpg" width="600px" />|<img src="./demo/hall.jpg" width="600px" />|

2. I-House's library:

|PartA (manual labeling)|PartB (automatic pairing)|
|---|---|
|<img src="./demo/lib_merged.jpg" width="600px" />|<img src="./demo/lib.jpg" width="600px" />|

3. Lower sproul plaza:

|PartA (manual labeling)|PartB (automatic pairing)|
|---|---|
|<img src="./demo/plaza_merged.jpg" width="600px" />|<img src="./demo/plaza.jpg" width="600px" />|
