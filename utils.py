import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class TrainDataset(Dataset):
    def __init__(self):
        train = np.loadtxt("data_train.txt", delimiter=',', dtype=int)
        self.x_data = torch.from_numpy(train[:, 0:-1])
        self.y_data = torch.from_numpy(train[:, [-1]])
        self.len = train.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        train = np.loadtxt("data_test.txt", delimiter=',', dtype=int)
        self.x_data = torch.from_numpy(train[:, 0:-1])
        self.y_data = torch.from_numpy(train[:, [-1]])
        self.len = train.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


def calc_loss(out, label):
    out = torch.sum(out, 1)
    p = torch.sum(label, 1)
    loss = p * -torch.log(out) + (1 - p) * -torch.log(1 - out)
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