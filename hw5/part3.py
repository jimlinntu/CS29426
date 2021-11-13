import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import random
import copy

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from scipy.stats import truncnorm

from model_part3 import Model, Model2, Model_res34, ModelUnet
from gaussian import paste_gaussian

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

matplotlib.use('Agg')

class Info():
    def __init__(self, folder, xml_name, is_test):
        assert isinstance(is_test, bool)
        self.folder = folder
        self.is_test = is_test
        self.paths, self.bboxes, self.landmarks = self.parse(folder / xml_name)
        self.n = len(self.paths)

    def get_path(self, idx):
        return str(self.folder / self.paths[idx])

    def parse(self, xml):
        tree = ET.parse(xml)
        root = tree.getroot()

        file_paths = []
        bboxes = [] # (top, left, h, w)
        landmarks = []

        for image in root[2]:
            file_path = image.attrib["file"]
            box_tag = image[0].attrib

            landmark = []
            if not self.is_test:
                for i in range(68):
                    x = int(image[0][i].attrib["x"])
                    y = int(image[0][i].attrib["y"])
                    landmark.append([x, y])

            file_paths.append(file_path)
            bboxes.append(
                    [box_tag["top"], box_tag["left"], box_tag["height"], box_tag["width"]])
            landmarks.append(landmark)

        return file_paths, np.array(bboxes, dtype=np.int32),\
                np.array(landmarks, dtype=np.int32)

    def scale_bbox(self, bbox, scale):
        top_left = np.array([bbox[1], bbox[0]])
        h, w = bbox[2], bbox[3]

        new_h, new_w = round(h * scale), round(w * scale)
        top_left -= np.array([new_w // 2 - w//2, new_h // 2 - h // 2])

        # (t, l, h, w)
        return np.array([top_left[1], top_left[0], new_h, new_w])

    def make_square(self, bbox):
        top_left = np.array([bbox[1], bbox[0]])
        h, w = bbox[2], bbox[3]

        if h == w:
            return bbox

        mx = max(h, w)
        new_h, new_w = mx, mx

        # Adjust the top_left
        if new_h > h:
            top_left[1] -= (new_h - h) // 2
        if new_w > w:
            top_left[0] -= (new_w - w) // 2
        return np.array([top_left[1], top_left[0], new_h, new_w])

    def visualize_random(self, out_folder):
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True)
        indices = random.sample(range(self.n), 5)

        for i in indices:
            name = self.paths[i]
            path = self.folder / name

            img = cv2.imread(str(path))

            bbox = self.bboxes[i]
            bbox = self.scale_bbox(bbox, 1.5)

            landmark = self.landmarks[i]
            if not self.is_test:
                img, landmark, bbox = random_flip(img, landmark, bbox)

            plt.imshow(img[:, :, [2,1,0]])
            ax = plt.gca()
            ax.add_patch(Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2],
                linewidth=1, edgecolor="r", facecolor="none"))

            # Draw the landmark
            for i, (x, y) in enumerate(landmark):
                plt.plot(x, y, 'bo', markersize=1)
                plt.text(x, y, str(i), color="red", fontsize=10)

            name_slash_to_underscore = name.replace("/", "_")
            plt.savefig(out_folder / "{}".format(name_slash_to_underscore))
            plt.close()
            plt.cla()
            plt.clf()

def visualize_landmark(img, landmark, path, vis_idx=False):
    plt.imshow(img[:, :, [2,1,0]])
    for i, (x, y) in enumerate(landmark):
        plt.plot(x, y, 'bo', markersize=1)
        if vis_idx:
            plt.text(x, y, str(i), color="red", fontsize=4)
    plt.savefig(path)
    plt.close()
    plt.cla()
    plt.clf()

def visualize_landmark_w_bbox(img, landmark, bbox, path):
    plt.imshow(img[:, :, [2,1,0]])
    for i, (x, y) in enumerate(landmark):
        plt.plot(x, y, 'bo', markersize=1)

    ax = plt.gca()
    ax.add_patch(Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2],
        linewidth=1, edgecolor="r", facecolor="none"))

    plt.savefig(path)
    plt.close()
    plt.cla()
    plt.clf()

def visualize_trunc_norm(path):
    Scale = get_truncated_normal(1.5, sd=0.2, low=1.3, upp=2.0)

    plt.hist(Scale.rvs(100000), density=True)

    plt.savefig(path)
    plt.close()
    plt.cla()
    plt.clf()

def visualize_heatmap(heatmap, path):
    out = (heatmap*255).astype(np.uint8)
    cv2.imwrite(path, out)

def color_jitter(img, cj):
    # (H, W, C) -> (C, H, W)
    transposed_img = np.transpose(img, [2, 0, 1])
    tensor = torch.from_numpy(transposed_img)
    jittered = cj(tensor)
    # (C, H, W) -> (H, W, C)
    return np.transpose(jittered.numpy(), [1, 2, 0])

def random_flip(img, landmark, bbox):
    # (H, W, 3)
    reindex = (
        np.array([17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                  27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                  28, 29, 30, 31,
                  36, 35, 34, 33, 32,
                  46, 45, 44, 43, 48, 47,
                  40, 39, 38, 37, 42, 41,
                  55, 54, 53, 52, 51, 50, 49,
                  60, 59, 58, 57, 56,
                  65, 64, 63, 62, 61,
                  68, 67, 66], dtype=np.int32)-1

    )

    h, w = img.shape[0:2]
    img = np.flip(img, axis=1)

    bh, bw = bbox[2], bbox[3]
    '''
        *-----* <--- become the new top left after flipping
        |     |
        |     |
        *-----*
    '''
    new_bbox = np.array([bbox[0], w-1-(bbox[1]+bw-1), bh, bw], dtype=np.int32)

    landmark[:, 0] = w - 1 - landmark[:, 0]
    landmark = landmark[reindex]
    return img, landmark, new_bbox

class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, info, type_, indices=None, heatmap=False):
        assert isinstance(info, Info)
        assert isinstance(type_, str)

        self.info = info
        if indices is None:
            self.indices = list(range(self.info.n))
        else:
            self.indices = indices
        self.input_size = (224, 224)

        self.type_ = type_
        self.heatmap = heatmap
        # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Scale = get_truncated_normal(1.5, sd=0.2, low=1.3, upp=1.7)
        self.Angle = get_truncated_normal(0, sd=5, low=-15, upp=15)

        # Cache the images
        self.imgs = [None for i in range(len(self.indices))]

        # https://pytorch.org/vision/stable/transforms.html?highlight=colorjitter
        self.cj = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def restore_landmark(landmark, M_inv):
        assert M_inv.shape == (2, 3)
        n = len(landmark)
        ones = np.ones((n, 1), dtype=np.float32)

        # (68, 3)
        homo_landmark = np.concatenate([landmark, ones], axis=1)

        # (68, 3) dot (3, 2) -> (68, 2)
        landmark_restored = homo_landmark.dot(M_inv.T)
        return np.rint(landmark_restored).astype(np.int32)

    def pipeline(self, img):
        assert isinstance(img, np.ndarray)
        # Resize it to self.input_size
        h, w = img.shape[0:2]
        bbox = np.array([0, 0, h, w])
        bbox = self.info.make_square(bbox)

        cropped, M_inv = self.crop(img, bbox, None)
        cropped_chw = np.transpose(cropped, [2, 0, 1])
        img_tensor = torch.from_numpy(cropped_chw).float()
        img_tensor = img_tensor[[2,1,0], :, :]
        img_tensor = img_tensor / 255.
        img_tensor = self.normalize(img_tensor)

        return img_tensor, M_inv

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        path = self.info.get_path(true_idx)
        # if self.imgs[idx] is None:
        img = cv2.imread(path)

        # Augment
        if self.type_ == "train":
            img = color_jitter(img, self.cj)
        # During the training time, we want our model to learn
        # to predict on different scale
        scale = 1.5 # in test time, we always choose scale=1.5

        # Augment
        # if self.type_ == "train":
        #     scale = self.Scale.rvs() # random scaling

        bbox = self.info.scale_bbox(self.info.bboxes[true_idx], scale)
        bbox = self.info.make_square(bbox)

        # Crucial to copy!
        landmark = copy.deepcopy(self.info.landmarks[true_idx])
        if self.type_ == "train" and random.random() < 0.5:
            # visualize_landmark(img, landmark, "vis.jpg")
            img, landmark, bbox = random_flip(img, landmark, bbox)

        # if self.type_ == "train":
        #     # angle = self.Angle.rvs() # sample an angle
        #     angle = 0.
        #     # Rotate w.r.t to the bbox center
        #     tl = np.array([bbox[1], bbox[0]])
        #     h, w = bbox[2], bbox[3]
        #     center = np.array([tl[0] + w // 2, tl[1] + h // 2], dtype=np.float32)
        #     img, landmark = self.rotate(img, landmark, center, angle)

        if not self.info.is_test:
            cropped, landmark, M_inv = self.crop(img, bbox, landmark)
        else:
            cropped, M_inv = self.crop(img, bbox, None)

        # cropped: (H, W, C)
        cropped_chw = np.transpose(cropped, [2, 0, 1])
        img_tensor = torch.from_numpy(cropped_chw).float()
        # (68, 2)
        if not self.info.is_test:
            if self.heatmap:
                keypoint_heatmaps = []
                for kp in landmark:
                    heatmap = paste_gaussian(kp, self.input_size[0], self.input_size[1])
                    if heatmap.sum() >= 0.0001:
                        heatmap = heatmap / heatmap.sum() # normalize
                    keypoint_heatmaps.append(heatmap)

                # (68, H, W)
                keypoint_heatmaps = np.array(keypoint_heatmaps, dtype=np.float32)
            else:
                landmark_tensor = torch.from_numpy(landmark).float()

        # BGR to RGB
        img_tensor = img_tensor[[2,1,0], :, :]
        # Normalize it to [0, 1]
        img_tensor = img_tensor / 255.
        # And apply 
        img_tensor = self.normalize(img_tensor)

        if not self.info.is_test:
            if self.heatmap:
                return img_tensor, torch.from_numpy(keypoint_heatmaps), M_inv
            else:
                return img_tensor, landmark_tensor / self.input_size[0], M_inv
        else:
            return img_tensor, M_inv

    def crop(self, img, bbox, landmark):
        top_left = np.array([bbox[1], bbox[0]])
        h, w = bbox[2], bbox[3]

        src_pts = np.array([top_left, [top_left[0]+w-1, top_left[1]],
                            [top_left[0], top_left[1]+h-1]])

        th, tw = self.input_size
        dst_pts = np.array([[0, 0], [tw-1, 0], [0, th-1]])

        M = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        cropped = cv2.warpAffine(img, M, (tw, th), borderValue=(0, 0, 0))

        if landmark is not None:
            ones = np.ones((landmark.shape[0], 1))
            # (68, 3) -> (68, 2)
            new_landmark = np.concatenate([landmark, ones], axis=1) @ (M.T)

        M_inv = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        if landmark is not None:
            return cropped, new_landmark, M_inv
        else:
            return cropped, M_inv

    def rotate(self, img, landmark, center, angle):
        h, w = img.shape[0:2]
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        img = cv2.warpAffine(img, M, (w, h))

        ones = np.ones((landmark.shape[0], 1))
        new_landmark = np.concatenate([landmark, ones], axis=1) @ (M.T)
        return img, new_landmark

class KeyPointDetector():
    def __init__(self, lr, heatmap):
        self.heatmap = heatmap
        if heatmap:
            self.model = ModelUnet().cuda()
            self.loss = torch.nn.KLDivLoss(reduction="sum")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = Model2().cuda()
            self.loss = torch.nn.L1Loss(reduction="sum")

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            # self.optimizer = torch.optim.Adam(
            #         [{"params": self.model.resnet.parameters()},
            #          {"params": self.model.fc.parameters(), "lr": lr}], lr=lr)

    def visualize_train_val_losses(self, train_losses, valid_losses, path):
        n = len(train_losses)
        x = np.arange(n)

        plt.title("training and validation losses for 68 keypoints detection".format())
        plt.plot(x, train_losses, "b", label="train losses")
        plt.plot(x, valid_losses, "g", label="valid losses")
        plt.xlabel("Epoch")
        if self.heatmap:
            plt.ylabel("KL Loss")
        else:
            plt.ylabel("MAE Loss")
        plt.legend()
        plt.savefig(path)

        plt.close()
        plt.cla()
        plt.clf()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def fit(self, trainset, validset, graph_path):
        if self.heatmap:
            self._fit_heatmap(trainset, validset, graph_path)
        else:
            self._fit_landmark(trainset, validset, graph_path)

    def _fit_landmark(self, trainset, validset, graph_path):

        tloader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=False)
        vloader = DataLoader(validset, batch_size=32, shuffle=False, drop_last=False)

        n_epochs = 100

        train_losses = []
        valid_losses = []
        true_valid_losses = []

        best_valid_loss = float("inf")
        best_true_valid_loss = float("inf")
        best_model = None

        try:
            t = trange(n_epochs)
            for epoch in t:
                # Train
                train_num_examples = 0
                train_loss = 0.

                self.model.train()
                for batch in tloader:
                    img, landmark, M_inv = batch

                    self.optimizer.zero_grad()

                    img = img.cuda()
                    landmark = landmark.cuda()
                    pred_landmark = self.model(img)

                    loss = self.loss(landmark, pred_landmark)

                    (loss / img.shape[0]).backward()
                    self.optimizer.step()

                    train_num_examples += img.shape[0]
                    train_loss += loss.detach().cpu().numpy()
                    # break

                train_loss = train_loss / train_num_examples
                train_losses.append(train_loss)
                # Valid
                self.model.eval()
                valid_num_examples = 0
                valid_loss = 0.
                true_valid_loss = 0.

                for batch in vloader:
                    img, landmark, M_inv = batch
                    img = img.cuda()
                    landmark = landmark.cuda()

                    with torch.no_grad():
                        pred_landmark = self.model(img)

                    loss = self.loss(landmark, pred_landmark)

                    valid_num_examples += img.shape[0]
                    valid_loss += loss.detach().cpu().numpy()

                    landmark = landmark.cpu().numpy() * 224
                    pred_landmark = pred_landmark.cpu().numpy() * 224
                    M_inv = M_inv.cpu().numpy()

                    for l, p, m in zip(landmark, pred_landmark, M_inv):
                        l = KaggleDataset.restore_landmark(l, m)
                        p = KaggleDataset.restore_landmark(p, m)
                        # (68, 2) <-> (68, 2)
                        true_valid_loss += np.sum(np.abs(l - p))
                    # break

                valid_loss = valid_loss / valid_num_examples
                valid_losses.append(valid_loss)

                true_valid_loss = true_valid_loss / (valid_num_examples * 68 * 2)
                true_valid_losses.append(true_valid_loss)

                # Save the model that performs the best on the validation set
                # if valid_loss < best_valid_loss:
                if true_valid_loss < best_true_valid_loss:
                    best_true_valid_loss = true_valid_loss
                    best_model = copy.deepcopy(self.model)
                    print("[!] New better valid loss: {} at epoch: {}".format(best_true_valid_loss, epoch))

                t.set_postfix(train_loss=train_loss, valid_loss=valid_loss, true_valid_loss=true_valid_loss)
        except KeyboardInterrupt:
            l = min(len(train_losses), len(valid_losses))
            # choose the shortest one
            train_losses = train_losses[:l]
            valid_losses = valid_losses[:l]

        # Visualize the train loss and valid losses
        self.visualize_train_val_losses(train_losses, valid_losses, graph_path)

        # Use the best model
        self.model = best_model

        if len(train_losses) > 0:
            print("[i] Training loss: {:f}".format(train_losses[-1]))
            print("[i] Validation loss: {:f}".format(valid_losses[-1]))
        print("[i] True Validation loss: {:f}".format(best_true_valid_loss))

    def _fit_heatmap(self, trainset, validset, graph_path):
        tloader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=False)
        vloader = DataLoader(validset, batch_size=32, shuffle=False, drop_last=False)

        n_epochs = 100

        train_losses = []
        valid_losses = []

        best_valid_loss = float("inf")
        best_model = None

        try:
            t = trange(n_epochs)
            for epoch in t:
                # Train
                train_num_batches = 0
                train_loss = 0.

                self.model.train()
                for batch in tloader:
                    img, heatmap, M_inv = batch
                    bsize = img.shape[0]

                    self.optimizer.zero_grad()

                    img = img.cuda()
                    # (N, 68, H, W)
                    heatmap = heatmap.cuda()
                    pred_heatmap = self.model(img)
                    # (N, 68, H, W) -> (N*68, H*W)
                    pred_heatmap = pred_heatmap.reshape(bsize*68, -1)
                    pred_heatmap = torch.nn.functional.log_softmax(pred_heatmap, dim=1)

                    heatmap = heatmap.reshape(bsize*68, -1)
                    pred_heatmap = pred_heatmap.reshape(bsize*68, -1)

                    # (N*68, H*W <--> (N*68, H*W))
                    loss = self.loss(pred_heatmap, heatmap) / (bsize * 68)

                    loss.backward()
                    self.optimizer.step()

                    train_num_batches += 1
                    train_loss += loss.detach().cpu().numpy()
                    # break

                train_loss = train_loss / train_num_batches
                train_losses.append(train_loss)
                # Valid
                self.model.eval()
                valid_num_batches = 0
                valid_loss = 0.

                for batch in vloader:
                    img, heatmap, M_inv = batch
                    bsize = img.shape[0]
                    img = img.cuda()
                    heatmap = heatmap.cuda()

                    with torch.no_grad():
                        pred_heatmap = self.model(img)
                        pred_heatmap = pred_heatmap.reshape(bsize*68, -1)
                        pred_heatmap = torch.nn.functional.log_softmax(pred_heatmap, dim=1)

                    loss = self.loss(pred_heatmap, heatmap.reshape(bsize*68, -1)) / (bsize * 68)

                    valid_num_batches += 1
                    valid_loss += loss.detach().cpu().numpy()
                    # break

                valid_loss = valid_loss / valid_num_batches
                valid_losses.append(valid_loss)

                # Save the model that performs the best on the validation set
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_model = copy.deepcopy(self.model)
                    print("[!] New better valid loss: {} at epoch: {}".format(best_valid_loss, epoch))

                t.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

        except KeyboardInterrupt:
            l = min(len(train_losses), len(valid_losses))
            # choose the shortest one
            train_losses = train_losses[:l]
            valid_losses = valid_losses[:l]

        # Visualize the train loss and valid losses
        self.visualize_train_val_losses(train_losses, valid_losses, graph_path)

        # Use the best model
        self.model = best_model

        if len(train_losses) > 0:
            print("[i] Training loss: {:f}".format(train_losses[-1]))
            print("[i] Validation loss: {:f}".format(valid_losses[-1]))

    def predict(self, testset, out_csv):
        if self.heatmap:
            self._predict_heatmap(testset, out_csv)
        else:
            self._predict_landmark(testset, out_csv)

    def _predict_landmark(self, testset, out_csv):
        assert isinstance(testset, KaggleDataset)
        assert isinstance(out_csv, str)
        self.model.eval()

        vis_folder = Path("./part3_pred/")
        vis_folder.mkdir(exist_ok=True)

        pred_landmarks = []

        # Sample some to visualize
        n_samples = 20
        vis_indices = random.sample(range(len(testset)), n_samples)
        vis_indices = set(vis_indices)

        self.model.eval()

        preds = []
        for i, item in enumerate(testset):
            img, M_inv = item

            # (C, H, W) -> (H, W, C)
            img = torch.unsqueeze(img, dim=0)
            img = img.cuda()

            with torch.no_grad():
                pred_landmark = self.model(img)

            pred_landmark = pred_landmark[0].cpu().numpy()
            pred_landmark = np.clip(pred_landmark, 0, 1)

            # (68, 2) * (1, 2)
            pred_landmark = pred_landmark * np.array(testset.input_size).reshape(1, 2)

            # Restore to the original coordinates
            pred_landmark = KaggleDataset.restore_landmark(pred_landmark, M_inv)

            preds.append(pred_landmark)
            # Visualize
            if i in vis_indices:
                img = cv2.imread(testset.info.get_path(testset.indices[i]))
                visualize_landmark(img, pred_landmark, vis_folder / f"{i}.jpg")

        # Write to the csv
        with open(out_csv, "w") as f:
            f.write("Id,Predicted\n")
            for i, landmark in enumerate(preds):
                # (68, 2) -> (68 * 2)
                flattened = landmark.flatten()
                for j, num in enumerate(flattened):
                    f.write("{},{}\n".format(i*68*2 + j, num))

    def _predict_heatmap(self, testset, out_csv):
        assert isinstance(testset, KaggleDataset)
        assert isinstance(out_csv, str)
        self.model.eval()

        vis_folder = Path("./part3_pred/")
        vis_folder.mkdir(exist_ok=True)

        pred_landmarks = []

        # Sample some to visualize
        n_samples = 20
        vis_indices = random.sample(range(len(testset)), n_samples)
        vis_indices = set(vis_indices)

        self.model.eval()

        preds = []
        for i, item in enumerate(testset):
            img, M_inv = item

            # (C, H, W) -> (H, W, C)
            img = torch.unsqueeze(img, dim=0)
            img = img.cuda()

            with torch.no_grad():
                pred_heatmap = self.model(img)[0]
                h, w = pred_heatmap.shape[1], pred_heatmap.shape[2]
                pred_heatmap = pred_heatmap.reshape(68, h*w)
                pred_heatmap = torch.nn.functional.softmax(pred_heatmap, dim=1)
                pred_heatmap = pred_heatmap.reshape(68, h, w)

            pred_heatmap = pred_heatmap.cpu().numpy()

            # (68, H, W) -> (68, 2)
            # Take the argument max for each heatmap
            shape = pred_heatmap.shape[1:3]
            pred_landmark = []
            for j in range(68):
                kp = np.unravel_index(pred_heatmap[j].argmax(), shape)
                pred_landmark.append([kp[1], kp[0]])

            pred_landmark = np.array(pred_landmark)
            # Restore to the original coordinates
            pred_landmark = KaggleDataset.restore_landmark(pred_landmark, M_inv)

            preds.append(pred_landmark)
            # Visualize
            if i in vis_indices:
                img = cv2.imread(testset.info.get_path(testset.indices[i]))
                visualize_landmark(img, pred_landmark, vis_folder / f"{i}.jpg")
                visualize_heatmap(pred_heatmap[0] / pred_heatmap[0].max(),
                        str(vis_folder / f"{i}_heatmap.jpg"))

        # Write to the csv
        with open(out_csv, "w") as f:
            f.write("Id,Predicted\n")
            for i, landmark in enumerate(preds):
                # (68, 2) -> (68 * 2)
                flattened = landmark.flatten()
                for j, num in enumerate(flattened):
                    f.write("{},{}\n".format(i*68*2 + j, num))

    def predict_single(self, img, testset, path):
        assert isinstance(img, np.ndarray)
        self.model.eval()

        img_tensor, M_inv = testset.pipeline(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        img_tensor = img_tensor.cuda()

        with torch.no_grad():
            pred_landmark = self.model(img_tensor)
        # (68, 2)
        pred_landmark = pred_landmark[0].cpu().numpy() * testset.input_size[0]
        pred_landmark = KaggleDataset.restore_landmark(pred_landmark, M_inv)

        visualize_landmark(img, pred_landmark, path)

def main():
    random.seed(678)
    torch.manual_seed(8902)
    np.random.seed(7890)
    torch.cuda.manual_seed(782)

    parser = argparse.ArgumentParser()
    parser.add_argument("out_csv", type=str)
    parser.add_argument("graph_path", type=str)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--heatmap", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--collect", type=str, default=None)
    parser.add_argument("--collect_out", type=str, default=None)
    args = parser.parse_args()

    folder = Path("./kaggle")

    train_info = Info(folder, "labels_ibug_300W_train.xml", is_test=False)
    test_info = Info(folder,  "labels_ibug_300W_test_parsed.xml", is_test=True)

    if False:
        train_info.visualize_random("./part3_vis/")
        test_info.visualize_random("./part3_vis_test/")
        # unit test
        print(train_info.make_square(np.array([0, 0, 20, 30])))
        print(train_info.make_square(np.array([0, 0, 30, 20])))

        train = KaggleDataset(train_info)

        img, landmark = train[7]
        visualize_landmark(img, landmark, "vis.jpg")
        visualize_landmark(cv2.imread(train_info.get_path(7)), train_info.landmarks[7], "vis_ori.jpg")

    # Split the training and validation
    ratio = 0.8

    n = train_info.n
    indices = list(range(n))

    random.shuffle(indices)

    split_idx = int(ratio * n)

    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    print("[i] # of train examples: {}".format(len(train_indices)))
    print("[i] # of valid examples: {}".format(len(valid_indices)))
    print("[i] # of testing examples: {}".format(test_info.n))

    train = KaggleDataset(train_info, "train", train_indices, args.heatmap)
    valid = KaggleDataset(train_info, "valid", valid_indices, args.heatmap)
    test = KaggleDataset(test_info, "test", indices=None, heatmap=args.heatmap)

    # Fit the training set
    print("[i] Lr: {}".format(args.lr))
    detector = KeyPointDetector(args.lr, args.heatmap)
    if not args.load and args.save:
        detector.fit(train, valid, args.graph_path)
        detector.save(args.save)
    else:
        detector.load(args.load)

    if False:
        # Just for testing
        valid.info.is_test = True
        detector.predict(valid, args.out_csv)
        return

    if False:
        train.info.is_test = True
        detector.predict(train, args.out_csv)
        return

    if False:
        visualize_trunc_norm("./trunc_norm_his.jpg")

    # Predict the testing set
    detector.predict(test, args.out_csv)

    # Predict my collection
    if args.collect and args.collect_out:
        folder = Path(args.collect)
        out_folder = Path(args.collect_out)
        out_folder.mkdir(exist_ok=True)
        for p in folder.iterdir():
            if p.name.endswith("jpeg"):
                img = cv2.imread(str(p))
                detector.predict_single(img, test, out_folder / p.name)

if __name__ == "__main__":
    main()
