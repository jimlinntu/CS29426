# Proj5 Facial Keypoint Detection with Neural Networks

* Name: Tzu-Chuan Lin
* SID: 3036360742
* Website: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj5/cs194-26-abu/>
* Email: tzu-chuan_lin@berkeley.edu

## Environment
* Python 3.7.11
* `conda install pytorch==1.8.0 torchvision==0.9.0  cudatoolkit=11.1 -c pytorch -c conda-forge`
* `pip install -r requirements.txt`

## Part 1

First, please `mkdir params/`

* SimpleModel:
    * `python main.py part1 SimpleModel --save params/simple.pt --out ./val_out/`

* SimpleModelDeeper:
    * `python main.py part1 SimpleModelDeeper --save params/deeper.pt --out ./val_out_deeper/`

* SimpleModelLargeKernel:
    * `python main.py part1 SimpleModelLargeKernel --save params/largekernel.pt --out val_out_largekernel/ --lr 0.0005`

