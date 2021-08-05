from utils import *
from dmr import DMR
import torch

from torch import load, sigmoid, cat, rand, bmm, mean, matmul
import torch.nn as nn
from torch.nn.init import uniform_
from torch.utils.data import Dataset,DataLoader

from torch.optim import Adam
from sys import argv
import numpy as np
import pickle
import time
from sklearn import preprocessing

my_config = {
    #"visual_features_dict": "E:/data/iqon/GPBPRcode/feat/visualfeatures",
    #"textural_idx_dict": "E:/data/iqon/GPBPRcode/feat/textfeatures",
    #"textural_embedding_matrix": "E:/data/iqon/GPBPRcode/feat/smallnwjc2vec"
    "visual_features_dict": "../process/item_visual_feature.npy",
    "textural_features_dict": "../process/item_text_feature.npy",
    "outfit_item_dict": "../process/outfit_item_new.pkl",
    "user_idx_dict": "../dmr/user_idx.pkl",
    "item_cate_dict": "../process/cate.npy",
    #"textural_embedding_matrix": "smallnwjc2vec",
    "model_file": r"./model_save/dmr14.model",
    "outfit_len_dict": "../dmr/outfit_len.pkl"
}

def load_embedding_weight(device):
    jap2vec = torch.load(my_config['textural_embedding_matrix'])
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight


class SigLoss(nn.Module):
    def __init__(self):
        super(SigLoss, self).__init__()

    def forward(self, out, label):
        out = torch.sum(out, 1)
        p = torch.sum(label, 1)
        loss = p*-torch.log(out+1e-8) + (1-p)*-torch.log(1-out+1e-8)
        loss = loss.mean()
        return loss


def trainning(model, train_data_loader, device, user_idx, opt):
    r"""
        using data from Args to train model

        Args:

            mode: -

            train_data_loader: mini-batch iteration

            device: device on which model train

            visual_features: look up table for item visual features

            text_features: look up table for item textural features

            opt: optimizer of model
    """
    model.train()
    model = model.to(device)
    criterion = SigLoss().to(device)
    stored_arr = []
    loss_sum = 0
    loss_arr = []
    auc_arr = []
    #time1 = time.time()
    for iteration, data in enumerate(train_data_loader):
        #time2 = time.time()
        #print('loader time:', time2-time1)
        #insert(0>1)
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask, his_outfit_cate, outfit_cate, targets = data

        #his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = Encoder(feature, outfit_item, text_features, visual_features, outfit_len)
        #assert(0>1)

        targets = targets.to(device)
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = his_outfit_text.to(device), his_outfit_visual.to(device), mask.to(device), outfit_text.to(device), outfit_visual.to(device), uid.to(device), outfit_his_mask.to(device), outfit_mask.to(device)
        his_outfit_cate, outfit_cate = his_outfit_cate.to(device), outfit_cate.to(device)
        output = model(his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, user_idx, outfit_his_mask, outfit_mask, his_outfit_cate, outfit_cate)

        loss = criterion(output, targets)
        iteration += 1
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_sum = loss_sum + loss.item()
        prob_1 = output[:, 0].tolist()
        # print(prob_1)
        target_1 = targets[:, 0].tolist()
        # print(target_1)
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

        if(iteration % 10) == 0:
            auc = calc_auc(stored_arr)
            print("iter",iteration)
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
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask, his_outfit_cate, outfit_cate, targets = data

        #his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = Encoder(features, outfit_item, text_features, visual_features, outfit_len)
        # assert(0>1)

        targets = targets.to(device)
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = his_outfit_text.to(
            device), his_outfit_visual.to(device), mask.to(device), outfit_text.to(device), outfit_visual.to(
            device), uid.to(device), outfit_his_mask.to(device), outfit_mask.to(device)
        his_outfit_cate, outfit_cate = his_outfit_cate.to(device), outfit_cate.to(device)

        it += 1

        with torch.no_grad():
            output = model(his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, user_idx, outfit_his_mask, outfit_mask, his_outfit_cate, outfit_cate)
            loss = calc_loss(output, targets)
            loss_sum += loss

            prob_1 = output[:, 0].tolist()
            # print(prob_1)
            target_1 = targets[:, 0].tolist()
            # print(target_1)
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])

    return loss_sum / it, calc_auc(stored_arr)


def F(batch_size, eb_size, device):
    print('loading  features')

    idx_file = open(my_config['user_idx_dict'], 'rb')
    user_idx = pickle.load(idx_file)
    user_idx = {key: torch.from_numpy(np.array(user_idx[key])).to(device) for key in user_idx}

    '''
    visual_file = open(my_config['visual_features_dict'], 'rb')
    visual_features = pickle.load(visual_file)
    visual_features['0'] = np.zeros((1,2048)).tolist()
    #visual_features['3875318'] = np.zeros((1,2048)).tolist()
    visual_features[0] = np.zeros((1,2048)).tolist()
    print('load v finish')
    #assert(0>1)

    text_file = open(my_config['textural_features_dict'], 'rb')
    text_features = pickle.load(text_file)
    text_features['0'] = [0]*300
    text_features[0] = [0]*300
    print('load t finish')
    '''

    visual_features = preprocessing.normalize(np.load(my_config['visual_features_dict']).reshape(-1, 2048), norm='l2')
    visual_features = visual_features.reshape(-1, 1, 2048)
    print('load v finish')

    text_features = preprocessing.normalize(np.load(my_config['textural_features_dict']), norm='l2')
    print('load t finish')

    item_cate = np.load(my_config['item_cate_dict'])
    print('load cate finish')

    oi_file = open(my_config['outfit_item_dict'], 'rb')
    outfit_item = pickle.load(oi_file)
    print('load o finish')

    outfit_len_file = open(my_config['outfit_len_dict'], 'rb')
    outfit_len = pickle.load(outfit_len_file)

    trainDataset = TrainDataset("../data/new_data_train1.txt", visual_features, text_features, outfit_item, outfit_len, item_cate)
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=4)

    testDataset = TrainDataset("../data/new_data_test1.txt", visual_features, text_features, outfit_item, outfit_len, item_cate)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4)

    '''
    for iteration, data in enumerate(train_loader):
        feature, targets = data
        features = Encoder(feature, outfit_item, text_features, visual_features)
        insert(0>1)
    '''

    try:
        print("loading model")
        dmr = load(my_config['model_file'], map_location=lambda x,y: x.cuda(2))
    except Exception as e:
        print(e)
        print('no module exists, created new one {}'.format(my_config['model_file']))

        dmr = DMR(batch_size=batch_size, eb_size=eb_size).to(device)

    opt = Adam([
        {
            'params': dmr.parameters(),
            'lr': 0.005,
        }
    ])

    #file1 = open('./res/loss_train', 'a')
    #file2 = open('./res/auc_train', 'a')
    #file3 = open('./res/loss_test', 'a')
    #file4 = open('./res/auc_test', 'a')
    for i in range(20):
        # 这里是单个进程的训练
        print("training---------------")
        print("epoch:", i)
        loss_a, auc_a = trainning(dmr, train_loader, device, user_idx, opt)
        #np.savetxt('loss'+str(i)+'.txt', np.array(loss_a))
        #np.savetxt('auc' + str(i) + '.txt', np.array(auc_a))
        #file1.write('\n'.join(str(loss_a)))
        #file2.write('\n'.join(str(auc_a)))


        #dmr.epoch += 1
        torch.save(dmr, './model_save/dmr'+str(i)+'.model')

        loss, auc = evaluating(dmr, test_loader, device, user_idx)
        print("evaluating: epoch:", i, "loss:", loss, "test_auc:", auc)
        #file3.write(str(loss)+'\n')
        #file4.write(str(auc)+'\n')

    #file1.close()
    #file2.close()
    #file3.close()
    #file4.close()


if __name__ == "__main__":

    # "cpu" or "cuda:x" x is GPU index like (0,1,2,3,)
    F(batch_size = 256, eb_size = 128, device = 'cuda:2')
    print('finish')




