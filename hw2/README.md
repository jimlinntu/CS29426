# Fun with Filters and Frequencies!

Name: Tzu-Chuan (Jim) Lin
SID: 3036360742
Email: tzu-chuan_lin@berkeley.edu
Link: 

## Part 1 - Fun with Filters

### Part 1.1: Finite Difference Operator & Part 1.2: Derivative of Gaussian (DoG) Filter

* `python main_part1.py`: It will generate:
    * `dimg_dx.png`
    * `dimg_dy.png`
    * `dimg_mag_vis.png`
    * `dimg_mag_vis_bin.png`
    * `dblur_dx.png`
    * `dblur_dy.png`
    * `dblur_dx_DoG.png`
    * `dblur_dy_DoG.png`

### Part 2.1: Image "Sharpening"

* `tai.jpg`
    * `python main_part2_1.py ./src_imgs/taj.jpg 1 taj_alpha1.jpg`
    * `python main_part2_1.py ./src_imgs/taj.jpg 2 taj_alpha2.jpg`
    * `python main_part2_1.py ./src_imgs/taj.jpg 3 taj_alpha3.jpg`

* `parkinglot.jpg`
    * `python main_part2_1.py --blur ./src_imgs/parkinglot.jpg 1 park_alpha_1.jpg`
    * `python main_part2_1.py --blur ./src_imgs/parkinglot.jpg 2 park_alpha_2.jpg`
    * `python main_part2_1.py --blur ./src_imgs/parkinglot.jpg 3 park_alpha_3.jpg`

### Part 2.2: Hybrid Images

* Derek + Nutmeg
    * `python main_part2_2.py --output_freq ./src_imgs/DerekPicture-crop.jpg ./src_imgs/nutmeg-crop.jpg 51 31 5`
* Parrot + Owl
    * `python main_part2_2.py --output_freq ./src_imgs/parrot-align.jpg ./src_imgs/owl-align.jpg 27 41 3`
* Nike + Adidas
    * `python main_part2_2.py --output_freq ./src_imgs/nike-align.jpeg ./src_imgs/adidas-align.jpeg 31 21 5`

### Part 2.3: Gaussian and Laplacian Stacks & Part 2.4: Multiresolution Blending (a.k.a. the oraple!)

* Apple + Orange
    * `python main_part2_34.py ./src_imgs/apple.jpeg ./src_imgs/orange.jpeg 4 71 21 21 ./out/`
* Car + Me
    * `python main_part2_34.py ./src_imgs/car-crop.jpeg ./src_imgs/Jim-crop.jpeg 4 101 13 13 ./out/`
* Parrot + Alpaca
    * `python main_part2_34.py ./src_imgs/parrot-front-crop.jpeg ./src_imgs/aplaca.jpeg --mask ./src_imgs/mask.jpeg 4 21 31 31 ./out/`
