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
from tqdm import tqdm
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

from model_part3 import Model, Model2

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

            plt.imshow(img[:, :, [2,1,0]])
            bbox = self.bboxes[i]
            bbox = self.scale_bbox(bbox, 1.5)
            ax = plt.gca()
            ax.add_patch(Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2],
                linewidth=1, edgecolor="r", facecolor="none"))

            # Draw the landmark
            landmark = self.landmarks[i]
            for (x, y) in landmark:
                plt.plot(x, y, 'bo', markersize=1)

            name_slash_to_underscore = name.replace("/", "_")
            plt.savefig(out_folder / "{}".format(name_slash_to_underscore))
            plt.close()
            plt.cla()
            plt.clf()

def visualize_landmark(img, landmark, path):
    plt.imshow(img[:, :, [2,1,0]])
    for (x, y) in landmark:
        plt.plot(x, y, 'bo', markersize=1)
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

def color_jitter(img):
    # https://pytorch.org/vision/stable/transforms.html?highlight=colorjitter
    cj = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)
    # (H, W, C) -> (C, H, W)
    transposed_img = np.transpose(img, [2, 0, 1])
    tensor = torch.from_numpy(transposed_img)
    jittered = cj(tensor)
    # (C, H, W) -> (H, W, C)
    return np.transpose(jittered.numpy(), [1, 2, 0])

class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, info, type_, indices=None):
        assert isinstance(info, Info)
        assert isinstance(type_, str)

        self.info = info
        if indices is None:
            self.indices = list(range(self.info.n))
        else:
            self.indices = indices
        self.input_size = (224, 224)

        self.type_ = type_
        # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Scale = get_truncated_normal(1.5, sd=0.2, low=1.3, upp=1.7)
        self.Angle = get_truncated_normal(0, sd=5, low=-15, upp=15)

        # Cache the images
        self.imgs = [None for i in range(len(self.indices))]

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
        return landmark_restored.astype(np.int32)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        path = self.info.get_path(true_idx)
        # if self.imgs[idx] is None:
        img = cv2.imread(path)
            # self.imgs[idx] = img
        # else:
        #     img = self.imgs[idx]

        # Augment
        if self.type_ == "train":
            img = color_jitter(img)
        # During the training time, we want our model to learn
        # to predict on different scale
        scale = 1.5 # in test time, we always choose scale=1.5

        # Augment
        if self.type_ == "train":
            scale = self.Scale.rvs() # random scaling

        bbox = self.info.scale_bbox(self.info.bboxes[true_idx], scale)
        bbox = self.info.make_square(bbox)

        landmark = self.info.landmarks[true_idx]

        if self.type_ == "train":
            angle = self.Angle.rvs() # sample an angle
            # Rotate w.r.t to the bbox center
            tl = np.array([bbox[1], bbox[0]])
            h, w = bbox[2], bbox[3]
            center = np.array([tl[0] + w // 2, tl[1] + h // 2], dtype=np.float32)
            img, landmark = self.rotate(img, landmark, center, angle)

        if not self.info.is_test:
            cropped, landmark, M_inv = self.crop(img, bbox, landmark)
        else:
            cropped, M_inv = self.crop(img, bbox, None)

        # cropped: (H, W, C)
        cropped_chw = np.transpose(cropped, [2, 0, 1])
        img_tensor = torch.from_numpy(cropped_chw).float()
        # (68, 2)
        if not self.info.is_test:
            landmark_tensor = torch.from_numpy(landmark).float()

        # BGR to RGB
        img_tensor = img_tensor[[2,1,0], :, :]
        # Normalize it to [0, 1]
        img_tensor = img_tensor / 255.
        # And apply 
        img_tensor = self.normalize(img_tensor)

        if not self.info.is_test:
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
        return img, landmark

class KeyPointDetector():
    def __init__(self, lr):
        self.model = Model2().cuda()
        self.loss = torch.nn.L1Loss(reduction="sum")

        self.optimizer = torch.optim.Adam(
                [{"params": self.model.resnet.parameters()},
                 {"params": self.model.fc.parameters(), "lr": lr}], lr=lr)

    def visualize_train_val_losses(self, train_losses, valid_losses, path):
        n = len(train_losses)
        x = np.arange(n)

        plt.title("training and validation losses for 68 keypoints detection".format())
        plt.plot(x, train_losses, "b", label="train losses")
        plt.plot(x, valid_losses, "g", label="valid losses")
        plt.xlabel("Epoch")
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

        tloader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=False)
        vloader = DataLoader(validset, batch_size=8, shuffle=False, drop_last=False)

        n_epochs = 100

        train_losses = []
        valid_losses = []

        best_valid_loss = float("inf")
        best_model = None

        for epoch in tqdm(range(n_epochs)):
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

            train_losses.append(train_loss / train_num_examples)
            # Valid
            self.model.eval()
            valid_num_examples = 0
            valid_loss = 0.

            for batch in vloader:
                img, landmark, M_inv = batch
                img = img.cuda()
                landmark = landmark.cuda()

                with torch.no_grad():
                    pred_landmark = self.model(img)

                loss = self.loss(landmark, pred_landmark)

                valid_num_examples += img.shape[0]
                valid_loss += loss.detach().cpu().numpy()
                # break

            valid_loss = valid_loss / valid_num_examples
            valid_losses.append(valid_loss)

            # Save the model that performs the best on the validation set
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)
                print("[!] New better valid loss: {} at epoch: {}".format(best_valid_loss, epoch))

        # Visualize the train loss and valid losses
        self.visualize_train_val_losses(train_losses, valid_losses, graph_path)

        # Use the best model
        self.model = best_model

        print("[i] Training loss: {:f}".format(train_losses[-1]))
        print("[i] Validation loss: {:f}".format(valid_losses[-1]))

    def predict(self, testset, out_csv):
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
            # [0, 224)
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

def main():
    random.seed(678)
    torch.manual_seed(8902)
    np.random.seed(7890)

    parser = argparse.ArgumentParser()
    parser.add_argument("out_csv", type=str)
    parser.add_argument("graph_path", type=str)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
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

    train = KaggleDataset(train_info, "train", train_indices)
    valid = KaggleDataset(train_info, "valid", valid_indices)
    test = KaggleDataset(test_info, "test", indices=None)

    # Fit the training set
    detector = KeyPointDetector(0.001)
    if not args.load and args.save:
        detector.fit(train, valid, args.graph_path)
        detector.save(args.save)
    else:
        detector.load(args.load)

    if False:
        # Just for testing
        train.info.is_test = True
        detector.predict(train)

    if False:
        visualize_trunc_norm("./trunc_norm_his.jpg")

    # Predict the testing set
    detector.predict(test, args.out_csv)

if __name__ == "__main__":
    main()
