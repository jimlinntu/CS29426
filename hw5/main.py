import cv2
import numpy as np
import matplotlib
import re
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from model import SimpleModel, SimpleModelDeeper, SimpleModelLargeKernel
import argparse
import random

from tqdm import tqdm

matplotlib.use('Agg')


def load_key_points(path):
    keypoints = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "#" in line:
                continue
            split = line.split("\t")
            if len(split) < 7:
                continue

            x, y = float(split[2]), float(split[3])
            keypoints.append([x, y])

    return np.array(keypoints)

def load_folder(folder):
    index_pat = re.compile("^[0-9][0-9]")
    folder = Path(folder)

    files = list(folder.iterdir())
    files.sort()

    imgs, keypoints = [], []
    val_imgs, val_keypoints = [], []
    img_names = []
    for p in files:
        if p.name.endswith(".jpg"):
            index = int(index_pat.search(p.name).group())

            key_point_path = p.with_suffix(".asf")
            img = cv2.imread(str(p))
            kp = load_key_points(key_point_path)

            if index <= 32:
                imgs.append(img)
                keypoints.append(kp)
            else:
                val_imgs.append(img)
                val_keypoints.append(kp)

            img_names.append(p.name)


    assert len(imgs) == len(keypoints)
    assert len(val_imgs) == len(val_keypoints)

    return (imgs, keypoints), (val_imgs, val_keypoints), img_names

def img2tensor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (80, 60))

    img = gray.astype(np.float32)

    img = img[np.newaxis, :, :]

    # normalize to [-0.5, 0.5]
    img = img / 255. - 0.5
    return torch.from_numpy(img)

def visualize(img, keypoints, path="vis.jpg"):
    h, w = img.shape[0:2]
    for i, p in enumerate(keypoints):
        x, y = p
        plt.plot(x*w, y*h, 'bo', markersize=4)
        plt.annotate("{}".format(i), (x*w, y*h), fontsize=12)

    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    plt.cla()
    plt.clf()

def visualize_w_gt(img, pred_keypoints, gt_keypoints, path):
    h, w = img.shape[0:2]
    for i, p in enumerate(gt_keypoints):
        x, y = p
        plt.plot(x*w, y*h, 'go', markersize=4)

    for i, p in enumerate(pred_keypoints):
        x, y = p
        plt.plot(x*w, y*h, 'ro', markersize=4)

    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    plt.cla()
    plt.clf()

class ImmDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, keypoints, part):
        assert isinstance(part, str)
        assert len(imgs) == len(keypoints)
        # Preload all things
        self.imgs = imgs
        self.keypoints = keypoints
        if part == "part1":
            # only leave the nose
            self.keypoints = [kp[52:53, :] for kp in self.keypoints]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = img2tensor(img)

        kp = self.keypoints[idx].astype(np.float32)
        kp = torch.from_numpy(kp)

        return img, kp

class Detector():
    def __init__(self, model_name, lr):
        self.model_name = model_name
        self.model = globals()[model_name](1).cuda()
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def visualize_train_val_losses(self, train_losses, valid_losses):
        n = len(train_losses)
        path = "loss_graph.jpg"
        x = np.arange(n)

        plt.title("{} training and validation losses".format(self.model_name))
        plt.plot(x, train_losses, "b", label="train losses")
        plt.plot(x, valid_losses, "g", label="valid losses")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(path)

        plt.close()
        plt.cla()
        plt.clf()

    def fit(self, trainset, validset, n_epochs=100):
        self.model.train()

        train_loader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
        valid_loader = DataLoader(validset, batch_size=4, shuffle=False, drop_last=False)

        train_losses = []
        valid_losses = []
        for epoch in tqdm(range(n_epochs)):
            train_num_examples = 0
            train_loss = 0.

            # Training loop
            for img, kp in train_loader:
                self.optimizer.zero_grad()

                batch_size = img.shape[0]

                img = img.cuda()
                kp = kp.cuda()

                # predicted keypoints
                pred_kp = self.model(img)

                loss = self.loss(kp, pred_kp)
                loss.backward()

                self.optimizer.step()

                train_num_examples += batch_size
                train_loss += loss.detach().cpu().numpy()

            train_losses.append(train_loss / train_num_examples)

            # Run the validation set
            valid_num_examples = 0
            valid_loss = 0.

            for img, kp in valid_loader:
                batch_size = img.shape[0]
                img = img.cuda()
                kp = kp.cuda()

                # predicted keypoints
                with torch.no_grad():
                    pred_kp = self.model(img)

                loss = self.loss(kp, pred_kp)
                valid_num_examples += batch_size
                valid_loss += loss.detach().cpu().numpy()

            valid_losses.append(valid_loss / valid_num_examples)

        # Visualize the train loss and valid losses
        self.visualize_train_val_losses(train_losses, valid_losses)

    def predict(self, img):
        assert isinstance(img, np.ndarray)
        self.model.eval()
        # during prediction, x, y will be inside 0, 1
        tensor_img = img2tensor(img).cuda()
        tensor_img = torch.unsqueeze(tensor_img, dim=0)
        kp = self.model(tensor_img)
        kp = torch.clip(kp, 0., 1.)

        # (1, num_keypoints, 2)
        kp = kp.detach().cpu().numpy()
        # (num_keypoints, 2)
        kp = kp[0]
        return kp

def main():
    torch.manual_seed(879)
    random.seed(798)
    parser = argparse.ArgumentParser()
    parser.add_argument("which_part", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if args.load == "" and args.save == "":
        return

    (imgs, keypoints), (val_imgs, val_keypoints), img_names\
        = load_folder("./imm_face")

    if False:
        indices = random.sample(range(len(imgs)), 3)
        for i in indices:
            print("Visualize: {}".format(img_names[i]))
            visualize(imgs[i], keypoints[i][52:53],
                      "nose_gt_{}".format(img_names[i]))

    if False:
        test_idx = 50
        trainset = ImmDataset(imgs[test_idx:test_idx+1],
                keypoints[test_idx:test_idx+1], part="part1")
        validset = ImmDataset(val_imgs, val_keypoints, part="part1")

        # Test fitting a single image
        detector = Detector()
        detector.fit(trainset, validset, n_epochs=100)

        pred_kp = detector.predict(imgs[0])
        print("pred_kp: {}".format(pred_kp))
        print("kp: {}".format(keypoints[test_idx][52]))
        visualize(imgs[test_idx], pred_kp)

    # Prepare the training set
    trainset = ImmDataset(imgs, keypoints, part=args.which_part)
    validset = ImmDataset(val_imgs, val_keypoints, part=args.which_part)

    detector = Detector(args.model_name, args.lr)
    if args.load == "":
        # Train the model
        detector.fit(trainset, validset, n_epochs=200)
        detector.save(args.save)
    else:
        detector.load(args.load)

    # Predict all valid images
    if args.which_part == "part1":
        out_folder = Path(args.out)
        out_folder.mkdir(exist_ok=True)
        for i, (img, kp) in tqdm(enumerate(zip(val_imgs, val_keypoints))):
            pred_kp = detector.predict(img)

            visualize_w_gt(img, pred_kp, kp[52:53],
                    out_folder / "{}.jpg".format(i))



if __name__ == "__main__":
    main()
