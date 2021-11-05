# Proj5 Facial Keypoint Detection with Neural Networks

* Name: Tzu-Chuan Lin
* SID: 3036360742
* Website: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj5/cs194-26-abu/>
* Email: tzu-chuan_lin@berkeley.edu

## Environment
* Python 3.7.11
* `conda install pytorch==1.8.0 torchvision==0.9.0  cudatoolkit=11.1 -c pytorch -c conda-forge`
* `pip install -r requirements.txt`
* Put the dataset into `./imm_face/`
* Download the models from [here](https://drive.google.com/drive/folders/1fxJXZc5fhexEvpC1qskAOpuYVTPHCzDP?usp=sharing) to `./params/`

## Part 1

* SimpleModel:
    * `python main.py part1 SimpleModel --load params/simple.pt --out ./val_out/`

* SimpleModelDeeper:
    * `python main.py part1 SimpleModelDeeper --load params/deeper.pt --out ./val_out_deeper/`

* SimpleModelLargeKernel:
    * `python main.py part1 SimpleModelLargeKernel --load params/largekernel.pt --out ./val_out_largekernel/`

## Part 2

* `Baseline`: `python part2.py Baseline ./base/ --load ./params/baseline.pt`
* `Baseline_3x3`: `python part2.py Baseline_3x3 ./base_3x3/ --load ./params/baseline_3x3.pt`
* `Baseline_5x5`: `python part2.py Baseline_5x5 ./base_5x5/ --load ./params/baseline_5x5.pt`
* `Baseline_7x7`: `python part2.py Baseline_7x7 ./base_7x7/ --load ./params/baseline_7x7.pt`
* `Baseline_9x9`: `python part2.py Baseline_9x9 ./base_9x9/ --load ./params/baseline_9x9.pt`

## Part 3
