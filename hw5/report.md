# Proj5 Facial Keypoint Detection with Neural Networks

Name: Tzu-Chuan Lin
## Part 1: Nose Tip Detection

In this section, I trained the models using three different architectures and without any data augmentation.

||SimpleModel|SimpleModel+one more conv layer<br>(SimpleModelDeeper)|SimpleModel with 5x5 filters<br>(SimpleModelLargeKernel)|
|---|---|---|---|
||5 Conv(3x3) + 2 FC|6 Conv(3x3) + 2 FC|5 Conv(5x5) + 2FC|

* Ground truth nose tip keypoints

<img src="./demo/nose_gt_12-5f.jpg" width="600px" />

<img src="./demo/nose_gt_18-4m.jpg" width="600px" />

<img src="./demo/nose_gt_19-2m.jpg" width="600px" />

* Training and validation losses:

<img src="./demo/loss_graph_simple_model.jpg" width="600px" />

<img src="./demo/loss_graph_simple_model_deeper.jpg" width="600px" />

<img src="./demo/loss_graph_simple_model_largekernel.jpg" width="600px" />

* Correct results (SimpleModel):

<img src="./demo/val_out/20.jpg" width="600px" />

<img src="./demo/val_out/24.jpg" width="600px" />

* Wrong results (SimpleModel):

<img src="./demo/val_out/17.jpg" width="600px" />

<img src="./demo/val_out/32.jpg" width="600px" />

Because I only trained `SimpleModel` by 192 images (without any augmentation),
the prediction seems more suspectible to the rotation of the face or change of expression.

## Part 2: Full Facial Keypoints Detection

## Part 3: Train With Larger Dataset
