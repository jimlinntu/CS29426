# Proj4B: Feature Matching for Autostitching

## Environment
* Python 3.6
* `pip install -r requirements.txt`

## How to Run the code

1. I-House's great hall:

```
python auto_stitch.py ./src_imgs/hall/mid.jpg hall.jpg --imgs ./src_imgs/hall/left.jpg ./src_imgs/hall/right.jpg
```

2. I-House's library:

```
python auto_stitch.py ./src_imgs/library/mid.jpg lib.jpg --imgs ./src_imgs/library/left.jpg ./src_imgs/library/right.jpg
```

3. Lower sproul plaza:

```
python auto_stitch.py ./src_imgs/plaza/mid.jpg plaza.jpg --imgs ./src_imgs/plaza/left.jpg ./src_imgs/plaza/right.jp
```

4. (B&W) 360 Cylindrical Panorama

Download the images to `./src_imgs/cylinder/` from [here](https://drive.google.com/drive/folders/1clK_1vjVdPHC_pYD4Lv5xABXUdHEB7PV?usp=sharing) and then run:

```
python cylindrical.py ./src_imgs/cylinder/ 1 360_sproul.jpg
```

5. (B&W) Rotational invariance descriptor

```
python auto_stitch.py ./src_imgs/library/mid.jpg lib-rot-inv.jpg --imgs ./src_imgs/library/left-rot.jpg ./src_imgs/library/right.jpg --ori
```

## Code Details for `auto_stitch.py`

* Detecting corner features in an image: see `HarrisCorner()`
* Implement Adaptive Non-Maximal Suppression: see `anms(...)`
* Extracting a Feature Descriptor for each feature point: see `get_feature_descriptors(...)`
* Matching these feature descriptors between two images: see `pair_corners(...)`
* Use a robust method (RANSAC) to compute a homography: see `ransac_mask(...)`
