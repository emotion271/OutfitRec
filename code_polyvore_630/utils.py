import numpy as np
import torch
from torch import cat
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import lmdb
from PIL import Image
import six
import pandas as pd
from torchvision import transforms


def open_lmdb(path):
    return lmdb.open(
        path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
    )


def read_image_list(image_list_fn):
    """Return image list for each fashion category.

    the image name of n-th item in c-th category in image_list[c][n]
    """
    files = image_list_fn
    return [open(fn).read().splitlines() for fn in files]


def load_semantic_data(semantic_fn):
    """Load semantic data."""
    data_fn = os.path.join(semantic_fn)
    with open(data_fn, "rb") as f:
        s2v = pickle.load(f)
    return s2v


def get_img_trans(phase, image_size=291):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if phase == "train":
        if image_size == 291:
            return transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif phase in ["test", "val"]:
        if image_size == 291:
            return transforms.Compose([transforms.ToTensor(), normalize])
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise KeyError


class Datum(object):
    """Abstract Class for Polyvore dataset."""

    def __init__(
        self,
        image_list,  # item list for each category
        variable_length=False,
        use_semantic=False,
        semantic=None,
        use_visual=False,
        image_dir="",
        lmdb_env=None,
    ):

        if variable_length:
            # regard the variable-length outfits as four-category
            self.cate_map = [0, 0, 1, 2]
            self.cate_name = ["top", "top", "bottom", "shoe"]
        else:
            # the normal outfits
            self.cate_map = [0, 1, 2]
            self.cate_name = ["top", "bottom", "shoe"]
        self.image_list = image_list
        self.use_semantic = use_semantic
        self.semantic = semantic
        self.use_visual = use_visual
        self.image_dir = image_dir
        self.lmdb_env = lmdb_env
        self.transforms = get_img_trans("test", 291)

    def load_image(self, c, n):
        """PIL loader for loading image.

        Return
        ------
        img: the image of n-th item in c-the category, type of PIL.Image.
        """
        img_name = self.image_list[c][n]
        # read with lmdb format
        if self.lmdb_env:
            with self.lmdb_env.begin(write=False) as txn:
                imgbuf = txn.get(img_name.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        else:
            # read from raw image
            path = os.path.join(self.image_dir, img_name)
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        return np.array(img)

    def load_semantics(self, c, n):
        """Load semantic embedding.

        Return
        ------
        vec: the semantic vector of n-th item in c-the category,
            type of torch.FloatTensor.
        """
        img_name = self.image_list[c][n]
        vec = self.semantic[img_name]
        return torch.from_numpy(vec.astype(np.float32))

    def semantic_data(self, indices):
        """Load semantic data of one outfit."""
        vecs = []
        # for simplicity, fill up top item for variable-length outfit
        if indices[1] == -1:
            indices[1] = indices[0]
        for idx, cate in zip(indices, self.cate_map):
            v = self.load_semantics(cate, idx)
            vecs.append(v)
        return vecs

    def visual_data(self, indices):
        """Load image data of ne outfit."""
        images = []
        # for simplicity, fill up top item for variable-length outfit
        if indices[1] == -1:
            indices[1] = indices[0]
        for idx, cate in zip(indices, self.cate_map):
            img = self.load_image(cate, idx)
            img = self.transforms(img)
            images.append(img)
        return images

    def get(self, tpl):
        """Convert a tuple to torch.FloatTensor."""
        if self.use_semantic and self.use_visual:
            tpl_s = self.semantic_data(tpl)
            tpl_v = self.visual_data(tpl)
            return tpl_v, tpl_s
        if self.use_visual:
            return self.visual_data(tpl)
        if self.use_semantic:
            return self.semantic_data(tpl)
        return tpl


class PolyvoreDataset(Dataset):
    def __init__(self, path):
        self.list_fmt = "image_list_{}"
        image_list = read_image_list([os.path.join('../FHN/data/polyvore/tuples_630', self.list_fmt.format(p)) for p in ['top', 'bottom', 'shoe']])
        # 由于train.csv中的top,bottom,shoe都是以图片id的形式表示的，通过image_list得到图片id与图片名称的对应关系，image_list[c][n]表示第c类种第n张图片的名称。
        semantic = load_semantic_data("../FHN/data/polyvore/sentence_vector/semantic.pkl")
        # 读取item的文本特征，N*D，N个item,文本特征为D维，2400维
        lmdb_env = open_lmdb("../FHN/data/polyvore/images/lmdb")
        # lmdb格式将每张图片预先读取，以291*291*3的三通道矩阵存储，如果不用lmdb，将lmdb_env=None, load_image方法中会读取原始图片
        self.datum = Datum(
            image_list,
            variable_length=False,
            use_semantic=True,
            semantic=semantic,
            use_visual=True,
            image_dir="../FHN/data/polyvore/images/291*291",
            lmdb_env=lmdb_env,
        )
        self.image_list = image_list
        self.user_his = np.array(pd.read_csv('./data/u_pre.csv', header=None, usecols=[2, 3, 4], dtype=np.int)).reshape(-1,50,3)
        # 读取用户历史数据,630个用户，每个用户50条，[630*50,3]reshape为[630,50,3]
        self.data_tpl = np.array(pd.read_csv(path, header=None, usecols=[1, 2, 3, 4, 5], dtype=np.int))
        # 读取train.csv或test.csv，[N,5],N为样本数量，5维分别为user,top,bottom,shoe,target
        self.uidxs = self.data_tpl[:, 0]
        self.outfit = self.data_tpl[:, 1:-1]
        self.target = self.data_tpl[:, -1]
        self.len = len(self.outfit)

    def __getitem__(self, index):
        uidx = self.uidxs[index]
        outfit = self.outfit[index]
        user_his = self.user_his[uidx]
        user_h = [self.datum.get(h) for h in user_his]
        target = self.target[index]

        return uidx, user_h, self.datum.get(outfit), target

    def __len__(self):
        return self.len


def calc_loss(out, label):
    out = torch.sum(out, 1)
    p = torch.sum(label, 1)
    loss = p * -torch.log(out+1e-8) + (1 - p) * -torch.log(1 - out+1e-8)
    loss = loss.mean()
    return loss


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc
