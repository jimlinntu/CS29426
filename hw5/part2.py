import cv2
import numpy as np
import matplotlib
import re
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import argparse
import random

from tqdm import tqdm

from model import Baseline, Baseline_5x5, Baseline_7x7, Baseline_9x9

class ImmDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, keypoints):
        assert len(imgs) == len(keypoints)
        # Preload all things
        self.imgs = imgs
        self.keypoints = keypoints

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # TODO: 
        img = self.imgs[idx].astype(np.uint8)
        img = preprocess(img)
        img = torch.from_numpy(img)

        kp = self.keypoints[idx].astype(np.float32)
        kp = torch.from_numpy(kp)

        img, kp = transform(img, kp)

        img = normalize(img)

        return img, kp

def normalize(img):
    return (img / 255.0) - 0.5

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (240, 180))
    gray = gray.astype(np.int32)
    return gray[np.newaxis, :, :]

def transform(img, kp):
    h, w = img.shape[1:3]
    num_keypoints = kp.shape[0]
    img = torchvision.transforms.ColorJitter(brightness=0.5)(img)

    angle = random.randint(-15, 15)
    rot_mat = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0).astype(np.float32)
    rot_mat = torch.from_numpy(rot_mat)

    img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR,
            expand=False)

    # (58, 2) -> (58, 3) dot (3, 2) -> (58, 2)
    kp = kp.type(torch.FloatTensor)
    kp = kp * torch.tensor([[w, h]])
    ones = torch.ones((num_keypoints, 1), dtype=torch.float32)
    kp = torch.cat([kp, ones], dim=1) @ (rot_mat.T)

    # Random shift
    shift_vec = torch.tensor([random.randint(-10, 10), random.randint(-10, 10)])

    img = TF.affine(img, angle=0, translate=shift_vec.tolist(), scale=1, shear=1)

    kp = kp + shift_vec.reshape(1, 2)

    kp = kp / torch.tensor([[w, h]])
    return img, kp

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

class Detector():
    def __init__(self, model_name, lr):
        self.model_name = model_name
        self.model = globals()[model_name]().cuda()
        print("===========================")
        print(self.model)
        print("===========================")
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

        plt.title("{} training and validation losses for 58 keypoints detection".format(self.model_name))
        plt.plot(x, train_losses, "b", label="train losses")
        plt.plot(x, valid_losses, "g", label="valid losses")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(path)

        plt.close()
        plt.cla()
        plt.clf()

    def visualize_filter(self, title, filters, path):
        assert isinstance(filters, np.ndarray)
        assert len(filters.shape) == 3

        num_filters = filters.shape[0]

        ncols = 8
        fig, ax = plt.subplots(nrows=num_filters // 8, ncols=ncols)
        fig.suptitle(title)

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                f = filters[i*ncols+j]
                normalized_f = cv2.normalize(f, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                normalized_f = normalized_f.astype(np.int32)
                col.axis("off")
                col.imshow(normalized_f, cmap="gray", vmin=0, vmax=255)

        plt.savefig(path)

        plt.close()
        plt.cla()
        plt.clf()

    def visualize_filters(self, folder):
        if self.model_name == "Baseline_7x7":
            Path(folder).mkdir(exist_ok=True)
            conv1_filters = self.model.conv1.weight.cpu().detach().numpy()
            conv1_filters = conv1_filters[:, 0, :, :]

            conv2_filters = self.model.conv2.weight.cpu().detach().numpy()
            conv2_filters = conv2_filters[:, 0, :, :]

            conv3_filters = self.model.conv3.weight.cpu().detach().numpy()
            conv3_filters = conv3_filters[:, 0, :, :]

            self.visualize_filter("{}'s conv1 filters".format(self.model_name),
                    conv1_filters, Path(folder) / "conv1.jpg")

            self.visualize_filter("{}'s conv2 filters".format(self.model_name),
                    conv3_filters, Path(folder) / "conv2.jpg")

            self.visualize_filter("{}'s conv3 filters".format(self.model_name),
                    conv3_filters, Path(folder) / "conv3.jpg")



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
            for i, (img, kp) in enumerate(train_loader):
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

        print("[i] Training loss: {:f}".format(train_losses[-1]))
        print("[i] Validation loss: {:f}".format(valid_losses[-1]))

    def predict(self, img):
        assert isinstance(img, np.ndarray)
        self.model.eval()
        # during prediction, x, y will be inside 0, 1
        img = preprocess(img)
        img = torch.from_numpy(img)
        img = normalize(img).cuda()

        img = torch.unsqueeze(img, dim=0)
        kp = self.model(img)
        kp = torch.clip(kp, 0., 1.)

        # (1, num_keypoints, 2)
        kp = kp.detach().cpu().numpy()
        # (num_keypoints, 2)
        kp = kp[0]
        return kp

def visualize(img, keypoints, path):
    h, w = img.shape[0:2]
    for i, p in enumerate(keypoints):
        x, y = p
        plt.plot(x*w, y*h, 'bo', markersize=4)

    if img.shape[2] == 3:
        plt.imshow(img[:, :, [2, 1, 0]])
    else:
        plt.imshow(img[:, :, 0], cmap="gray", vmin=0, vmax=255)
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


def sample_and_visualize(imgs, keypoints, img_names):
    # Sample several images
    indices = random.sample(range(len(imgs)), 5)
    for i in indices:
        img = imgs[i]
        kp = keypoints[i]
        img = preprocess(img)
        img, kp = transform(torch.from_numpy(img), torch.from_numpy(kp))
        # (1, h, w)
        img = img.numpy().astype(np.int32)
        # (h, w, 1)
        img = np.transpose(img, [1, 2, 0])
        img = np.tile(img, [1, 1, 3])
        kp = kp.numpy()
        visualize(img, kp, "./demo/part2/{}".format(img_names[i]))
    return

def main():
    torch.manual_seed(879)
    random.seed(798)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("out", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    (imgs, keypoints), (val_imgs, val_keypoints), img_names\
        = load_folder("./imm_face")

    trainset = ImmDataset(imgs, keypoints)
    validset = ImmDataset(val_imgs, val_keypoints)

    if False:
        sample_and_visualize(imgs, keypoints, img_names)
        return

    print("[*] Using model: {}".format(args.model_name))

    if args.load is None:
        detector = Detector(args.model_name, 0.001)
        detector.fit(trainset, validset, n_epochs=100)

        if args.save:
            detector.save(args.save)
    else:
        detector = Detector(args.model_name, 0.001)
        detector.load(args.load)

    out_folder = Path(args.out)
    out_folder.mkdir(exist_ok=True)
    for i, (img, kp) in tqdm(enumerate(zip(val_imgs, val_keypoints))):
        pred_kp = detector.predict(img)

        visualize_w_gt(img, pred_kp, kp,
                out_folder / "{}.jpg".format(i))

    # Visualize the learned filters
    detector.visualize_filters("./demo/learned_filters/")


if __name__ == "__main__":
    main()
