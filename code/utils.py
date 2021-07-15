import numpy as np
import torch
from torch import cat
from torch.utils.data import Dataset,DataLoader
import time


class TrainDataset(Dataset):
    def __init__(self, visual_feature, text_feature, outfit_item, outfit_len):
        train = np.loadtxt("../data/new_data_train.txt", delimiter=',', dtype=int)
        self.x_data = torch.from_numpy(train[:, 0:-1])
        self.y_data = torch.from_numpy(train[:, [-1]])
        self.visual_features = visual_feature
        self.text_features = text_feature
        self.outfit_item = outfit_item
        self.outfit_len = outfit_len
        self.len = train.shape[0]

    def __getitem__(self, index):
        feature = self.x_data[index]
        target = self.y_data[index]
        visual_features = self.visual_features
        text_features = self.text_features
        outfit_item = self.outfit_item
        outfit_len = self.outfit_len

        his_outfit = feature[:50]
        mask = feature[50:100]
        outfit = feature[100:101]
        #print(outfit)
        uid = feature[101:102]
        #time1 = time.time()
        his_outfit_text = cat([cat([torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in his_outfit], 0)
        #time2 = time.time()
        #print('cat text time:', time2-time1)
        #for outfit in his_outfit:
            #for items in outfit_item[outfit.item()]:
                #time3 = time.time()
                #a = text_features[items]
                #time4 = time.time()
        #time5 = time.time()
        #print('one search:', time4-time3)
        #print('for time:', time5-time2)
        #assert(0>1)
        #time6 = time.time()
        his_outfit_visual = cat([cat([torch.Tensor(visual_features[items]).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in his_outfit], 0)
        #time7 = time.time()
        #print('cat img time:', time7-time6)
        outfit_text = cat([torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) 
        outfit_visual = cat([torch.Tensor(visual_features[items]).unsqueeze(0) for items in outfit_item[outfit.item()]], 1)

        outfit_his_mask = cat([torch.Tensor([1] * outfit_len[outfit.item()] + [0] * (20 - outfit_len[outfit.item()])).unsqueeze(0) for outfit in his_outfit], 0)
        outfit_mask = torch.Tensor([1] * outfit_len[outfit.item()] + [0] * (20 - outfit_len[outfit.item()])).unsqueeze(0)
        #time8 = time.time()
        #print('all time:', time8-time1)

        return his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask, target

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self, visual_feature, text_feature, outfit_item, outfit_len):
        train = np.loadtxt("../data/new_data_test.txt", delimiter=',', dtype=int)
        self.x_data = torch.from_numpy(train[:, 0:-1])
        self.y_data = torch.from_numpy(train[:, [-1]])
        self.visual_features = visual_feature
        self.text_features = text_feature
        self.outfit_item = outfit_item
        self.outfit_len = outfit_len
        self.len = train.shape[0]

    def __getitem__(self, index):
        feature = self.x_data[index]
        target = self.y_data[index]
        visual_features = self.visual_features
        text_features = self.text_features
        outfit_item = self.outfit_item
        outfit_len = self.outfit_len

        his_outfit = feature[:50]
        mask = feature[50:100]
        outfit = feature[100:101]
        uid = feature[101:102]
        his_outfit_text = cat([cat(
            [torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) for items in outfit_item[outfit.item()]], 1)
                               for outfit in his_outfit], 0)
        his_outfit_visual = cat(
            [cat([torch.Tensor(visual_features[items]).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for
             outfit in his_outfit], 0)
        outfit_text = cat(
            [torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) for items in outfit_item[outfit.item()]], 1)
        outfit_visual = cat([torch.Tensor(visual_features[items]).unsqueeze(0) for items in outfit_item[outfit.item()]],1)

        outfit_his_mask = cat(
            [torch.Tensor([1] * outfit_len[outfit.item()] + [0] * (20 - outfit_len[outfit.item()])).unsqueeze(0) for
             outfit in his_outfit], 0)
        outfit_mask = torch.Tensor([1] * outfit_len[outfit.item()] + [0] * (20 - outfit_len[outfit.item()])).unsqueeze(0)


        return his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask, target

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
