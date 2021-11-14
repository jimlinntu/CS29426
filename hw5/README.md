# Proj5 Facial Keypoint Detection with Neural Networks

* Name: Tzu-Chuan Lin
* SID: 3036360742
* Website: <https://inst.eecs.berkeley.edu/~cs194-26/fa21/upload/files/proj5/cs194-26-abu/>
* Email: tzu-chuan_lin@berkeley.edu
* My Kaggle email: jimlin7777@hotmail.com

## Environment
* Python 3.7.11
* `conda install pytorch==1.8.0 torchvision==0.9.0  cudatoolkit=11.1 -c pytorch -c conda-forge`
* `pip install -r requirements.txt`
* Put the [imm dataset](https://web.archive.org/web/20210305094647/http://www2.imm.dtu.dk/~aam/datasets/datasets.html) into `./imm_face/`
* Download the models from [here](https://drive.google.com/drive/folders/1fxJXZc5fhexEvpC1qskAOpuYVTPHCzDP?usp=sharing) to `./params/`
* Place the kaggle dataset from [here](https://www.kaggle.com/c/cs194-26-fall-2021-project-5/data) at `./kaggle/` like this:
    ```
    ./kaggle/
    ├── afw
    ├── helen
    ├── ibug
    ├── labels_ibug_300W_test_parsed.xml
    ├── labels_ibug_300W_train.xml
    └── lfpw
    ```
* `mkdir ./out/`

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

* Train the neural network:
```
python part3.py ./out/out.csv ./loss_graph_model2_mae_w_jitter_flip.jpg --save ./params/resnet_kp_model2_jitter_flip.pt
```

* Predict the testing set and the 3 photos I collected (using `ResNet18 wo augmentation`)
```
python part3.py ./out/out.csv ./loss_graph.jpg --load ./params/resnet_kp_model2_mae.pt --collect ./collect/ --collect_out ./collect_out/
```

* Predict the testing set and the 3 photos I collected (using `ResNet18 w augmentation`):
```
python part3.py ./out/out.csv ./loss_graph_model2_mae_w_jitter_flip.jpg --load ./params/resnet_kp_model2_jitter_flip.pt --collect ./collect/ --collect_out ./collect_out/
```

## B&W
* Train the keypoint gaussian model (predict a `(68, h, w)` map)
```
python part3.py ./out/out.csv ./loss_graph_heat.jpg --save ./params/resnet_kp_model_heat.pt --heatmap
```

* Predict
```
python part3.py ./out/out.csv ./loss_graph_heat.jpg --load ./params/resnet_kp_model_heat.pt --heatmap
```

