from model.dmr import DMR
from model.utils import *
import torch

import torch.nn as nn
from torch.nn.init import uniform_
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from sys import argv
import numpy as np
import pickle
import time
from sklearn import preprocessing

my_config = {
    "model_file": r"./model_save_sgd/dmr19.model"
}

device_ids = [1, 2, 3]


def to_device(data, device):
    """Move data to device."""
    from collections import Sequence

    error_msg = "data must contain tensors or lists; found {}"
    if isinstance(data, Sequence):
        return tuple(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    raise TypeError((error_msg.format(type(data))))


class SigLoss(nn.Module):
    def __init__(self):
        super(SigLoss, self).__init__()

    def forward(self, out, label):
        out = torch.sum(out, 1)
        p = torch.sum(label, 1)
        loss = p*-torch.log(out+1e-8) + (1-p)*-torch.log(1-out+1e-8)
        loss = loss.mean()
        return loss


def trainning(model, train_data_loader, device, opt):
    model.train()
    model = model.to(device)
    criterion = SigLoss().to(device)
    stored_arr = []
    loss_sum = 0
    loss_arr = []
    auc_arr = []
    for iteration, data in enumerate(train_data_loader):
        uid, user_his, outfit, target = data
        uid = to_device(uid, device)
        user_his = to_device(user_his, device)
        outfit = to_device(outfit, device)
        output = model(uid, user_his, outfit)
        target = target.view(-1, 1)
        target = to_device(target, device)
        loss = criterion(output, target)
        iteration += 1
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_sum = loss_sum + loss.item()
        prob_1 = output[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

        if(iteration % 10) == 0:
            auc = calc_auc(stored_arr)
            print("iter", iteration)
            print("auc------------------", auc)
            print("loss-----------------", loss_sum / 10)
            loss_arr.append(loss_sum / 10)
            auc_arr.append(auc)
            stored_arr = []
            loss_sum = 0.

    return loss_arr, auc_arr


def evaluating(model, test_loader, device, user_idx):

    model.eval()
    loss_sum = 0.
    stored_arr = []
    it = 0
    for iteration, data in enumerate(test_loader):
        uid, user_his, outfit, target = data
        uid = to_device(uid, device)
        user_his = to_device(user_his, device)
        outfit = to_device(outfit, device)
        target = target.view(-1, 1)
        target = to_device(target, device)

        it += 1

        with torch.no_grad():
            output = model(uid, user_his, outfit)
            loss = calc_loss(output, target)
            loss_sum += loss

            prob_1 = output[:, 0].tolist()
            # print(prob_1)
            target_1 = target[:, 0].tolist()
            # print(target_1)
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])

    return loss_sum / it, calc_auc(stored_arr)


def run(batch_size, eb_size, device):

    trainDataset = PolyvoreDataset('./data/train.csv')
    train_loader = DataLoader(trainDataset, batch_size=batch_size*len(device_ids), shuffle=True, drop_last=True, num_workers=6)

    testDataset = PolyvoreDataset('./data/test.csv')
    test_loader = DataLoader(testDataset, batch_size=batch_size*len(device_ids), shuffle=False, num_workers=6)

    try:
        print("loading model")
        dmr = torch.load(my_config['model_file'], map_location=lambda x,y: x.cuda(2))
    except Exception as e:
        print(e)
        print('no module exists, created new one {}'.format(my_config['model_file']))

        dmr = DMR(batch_size=batch_size, eb_size=eb_size)
        dmr = torch.nn.DataParallel(dmr, device_ids=device_ids)

    '''
    opt = Adam([
        {
            'params': dmr.parameters(),
            'lr': 0.001,
        }
    ])
    '''
    opt = torch.optim.SGD(dmr.parameters(), lr=0.001, momentum=0.9)

    all_auc = []
    for i in range(60):
        # 这里是单个进程的训练
        print("training---------------")
        print("epoch:", i)
        loss_a, auc_a = trainning(dmr, train_loader, device, opt)

        dmr.epoch += 1
        torch.save(dmr, './model_save_sgd/dmr'+str(i)+'.model')

        loss, auc = evaluating(dmr, test_loader, device)
        print("evaluating: epoch:", i, "loss:", loss, "test_auc:", auc)
        all_auc.append(auc)
        print('all_auc:', all_auc)


if __name__ == "__main__":

    # "cpu" or "cuda:x" x is GPU index like (0,1,2,3,)
    run(batch_size = 6, eb_size = 128, device = device_ids[0])
    print('ffff')




