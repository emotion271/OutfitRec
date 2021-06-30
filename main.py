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

my_config = {
    #"visual_features_dict": "E:/data/iqon/GPBPRcode/feat/visualfeatures",
    #"textural_idx_dict": "E:/data/iqon/GPBPRcode/feat/textfeatures",
    #"textural_embedding_matrix": "E:/data/iqon/GPBPRcode/feat/smallnwjc2vec"
    "visual_features_dict": "../visual_features.pkl",
    "textural_features_dict": "../item_textfeature.pkl",
    "outfit_item_dict": "outfit_item_new_20.pkl",
    "user_idx_dict": "user_idx.pkl",
    #"textural_embedding_matrix": "smallnwjc2vec",
    "model_file": r"./model/dmr0.model",
    "outfit_len_dict": "outfit_len.pkl"
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
        loss = p*-torch.log(out) + (1-p)*-torch.log(1-out)
        loss = loss.mean()
        return loss

def Encoder(feature, outfit_item, text_features, visual_features, outfit_len):
    his_outfit  = feature[:, :50]
    mask = feature[:, 50:100]
    outfit = feature[:, 100:101]
    uid = feature[:, 101:102]
    #print('00000000000000000', visual_features['0'])
    #print('11111111111111', visual_features[0])
    #print(his_outfit)
    '''
    for I in his_outfit:
        print(I)
        for outfit in I:
            all_item = outfit_item[outfit.item()]
            print('all_item', all_item)
            outfit_text = mean(cat([torch.Tensor(visual_features[items]).unsqueeze(0) for items in all_item]), 0)
            print(outfit_text)
            print(outfit_text.size())
        x = cat([mean(cat([torch.Tensor(visual_features[items]).unsqueeze(0) for items in outfit_item[outfit.item()]]), 0) for outfit in I], 0)
        print(x.size())
    '''
    '''
    his_outfit_text = cat([cat([mean(cat([torch.Tensor(text_features[items]).unsqueeze(0) if items in text_features.keys() else torch.zeros(300).unsqueeze(0) for items in outfit_item[outfit.item()]]), 0).unsqueeze(0) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    print(his_outfit_text.size())
    tmp = cat([cat([cat([torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) if items in text_features.keys() else torch.zeros(300).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    print(tmp.size())
    his_outfit_visual = cat([cat([mean(cat([torch.Tensor(visual_features[items]).unsqueeze(0) if items in visual_features.keys() else torch.zeros((1, 2048)).unsqueeze(0) for items in outfit_item[outfit.item()]]), 0).unsqueeze(0) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    #his_outfit_visual = his_outfit_visual.squeeze(2)
    print(his_outfit_visual.size())
    tmp1 = cat([cat([cat([torch.Tensor(visual_features[items]).unsqueeze(0) if items in visual_features.keys() else torch.zeros((1, 2048)).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    print(tmp1.size())
    insert(0>1)
    outfit_text = cat([cat([mean(cat([torch.Tensor(text_features[items]).unsqueeze(0) if items in text_features.keys() else torch.zeros(300).unsqueeze(0) for items in outfit_item[outfit.item()]]), 0).unsqueeze(0) for outfit in I], 0).unsqueeze(0) for I in outfit], 0)
    #print(outfit_text.size())
    outfit_visual = cat([cat([mean(cat([torch.Tensor(visual_features[items]).unsqueeze(0) if items in visual_features.keys() else torch.zeros((1, 2048)).unsqueeze(0) for items in outfit_item[outfit.item()]]), 0).unsqueeze(0) for outfit in I], 0).unsqueeze(0) for I in outfit], 0)
    outfit_visual = outfit_visual.squeeze(2)
    #print(outfit_visual.size())
    '''

    his_outfit_text = cat([cat([cat([torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) if items in text_features.keys() else torch.zeros(300).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    his_outfit_visual = cat([cat([cat([torch.Tensor(visual_features[items]).unsqueeze(0) if items in visual_features.keys() else torch.zeros((1, 2048)).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    outfit_text = cat([cat([cat([torch.Tensor(text_features[items]).unsqueeze(0).unsqueeze(0) if items in text_features.keys() else torch.zeros(300).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in I], 0).unsqueeze(0) for I in outfit], 0)
    outfit_visual = cat([cat([cat([torch.Tensor(visual_features[items]).unsqueeze(0) if items in visual_features.keys() else torch.zeros((1, 2048)).unsqueeze(0) for items in outfit_item[outfit.item()]], 1) for outfit in I], 0).unsqueeze(0) for I in outfit], 0)
    #(his_outfit_text.size())
    #print(his_outfit_visual.size())
    #print(outfit_text.size())
    #print(outfit_visual.size())
    # assert(0>1)

    outfit_his_mask = cat([cat([torch.Tensor([1]*outfit_len[outfit.item()]+[0]*(20-outfit_len[outfit.item()])).unsqueeze(0) for outfit in I], 0).unsqueeze(0) for I in his_outfit], 0)
    outfit_mask = cat([cat([torch.Tensor([1] * outfit_len[outfit.item()] + [0] * (20 - outfit_len[outfit.item()])).unsqueeze(0) for outfit in I], 0).unsqueeze(0) for I in outfit], 0)
    #print(outfit_mask)

    return his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask





def trainning(model, mode, train_data_loader, device, visual_features, text_features, outfit_item, user_idx, outfit_len, opt):
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
    for iteration, data in enumerate(train_data_loader):
        feature, targets = data
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = Encoder(feature, outfit_item, text_features, visual_features, outfit_len)
        #assert(0>1)
        targets = targets.to(device)
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = his_outfit_text.to(device), his_outfit_visual.to(device), mask.to(device), outfit_text.to(device), outfit_visual.to(device), uid.to(device), outfit_his_mask.to(device), outfit_mask.to(device)
        output = model(his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, user_idx, outfit_his_mask, outfit_mask, weight=False)
        # print(output.size())
        # output = torch.sum(output, 1)
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


def evaluating(model, mode, test_loader, device, visual_features, text_features, outfit_item, user_idx, outfit_len):

    model.eval()
    loss_sum = 0.
    stored_arr = []
    it = 0
    for iteration, data in enumerate(test_loader):
        features, targets = data
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = Encoder(
            features, outfit_item, text_features, visual_features, outfit_len)
        # assert(0>1)
        targets = targets.to(device)
        his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, outfit_his_mask, outfit_mask = his_outfit_text.to(
            device), his_outfit_visual.to(device), mask.to(device), outfit_text.to(device), outfit_visual.to(
            device), uid.to(device), outfit_his_mask.to(device), outfit_mask.to(device)
        it += 1
        with torch.no_grad():
            output = model(his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, user_idx, outfit_his_mask, outfit_mask, weight=False)
            loss = calc_loss(output, targets)
            loss_sum += loss

            prob_1 = output[:, 0].tolist()
            # print(prob_1)
            target_1 = targets[:, 0].tolist()
            # print(target_1)
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])

    return loss_sum / it, calc_auc(stored_arr)

def F(mode ,hidden_dim, batch_size, eb_size, device):
    print('loading top&bottom features')

    #visual_features = torch.load(my_config['visual_features_dict'], map_location=lambda a, b: a.cpu())
    # print(len(visual_features))
    # visual_features['0'] = torch.zeros(2048)
    # print(len(visual_features))
    #text_features = torch.load(my_config['textural_idx_dict'], map_location=lambda a, b: a.cpu())
    #outfit_item = torch.load(my_config['outfit_item_dict'], map_location=lambda a, b: a.cpu())
    #print(text_features['12727752'])
    #assert(0>1)
    #text_features['0'] = (torch.ones(83) * 54275).to(int)
    # print(text_features['0'])
    #visual_features = {key: visual_features[key].cuda() for key in visual_features}
    #text_features = {key: text_features[key].cuda() for key in text_features}

    #user_idx = torch.load(my_config['user_idx_dict'], map_location=lambda a, b: a.cuda(1))
    idx_file = open(my_config['user_idx_dict'], 'rb')
    user_idx = pickle.load(idx_file)
    user_idx = {key: torch.from_numpy(np.array(user_idx[key])).cuda(1) for key in user_idx}

    visual_file = open(my_config['visual_features_dict'], 'rb')
    visual_features = pickle.load(visual_file)
    visual_features['0'] = np.zeros((1,2048)).tolist()
    visual_features[0] = np.zeros((1,2048)).tolist()
    #print(visual_features['33182241'])
    #print(visual_features['0'])
    print('load v finish')
    #assert(0>1)

    text_file = open(my_config['textural_features_dict'], 'rb')
    text_features = pickle.load(text_file)
    text_features['0'] = [0]*300
    text_features[0] = [0]*300
    print('load t finish')

    oi_file = open(my_config['outfit_item_dict'], 'rb')
    outfit_item = pickle.load(oi_file)
    print('load o finish')

    outfit_len_file = open(my_config['outfit_len_dict'], 'rb')
    outfit_len = pickle.load(outfit_len_file)

    trainDataset = TrainDataset()
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, drop_last=True)

    testDataset = TestDataset()
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    '''
    for iteration, data in enumerate(train_loader):
        feature, targets = data
        features = Encoder(feature, outfit_item, text_features, visual_features)
        insert(0>1)
    '''
    try:
        print("loading model")
        dmr = load(my_config['model_file'], map_location=lambda x,y: x.cuda(1))
    except Exception as e:
        print(e)
        print('no module exists, created new one {}'.format(my_config['model_file']))
        #embedding_weight = load_embedding_weight(device)
        dmr = DMR(batch_size=batch_size, eb_size=eb_size, uniform_value=0.3).to(device)

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
    for i in range(5):
        # 这里是单个进程的训练
        print("training---------------")
        print("epoch:", i)
        loss_a, auc_a = trainning(dmr, mode, train_loader, device, visual_features, text_features, outfit_item, user_idx, outfit_len, opt)
        #np.savetxt('loss'+str(i)+'.txt', np.array(loss_a))
        #np.savetxt('auc' + str(i) + '.txt', np.array(auc_a))
        #file1.write('\n'.join(str(loss_a)))
        #file2.write('\n'.join(str(auc_a)))


        dmr.epoch += 1
        torch.save(dmr, './model/dmr'+str(i)+'.model')

        loss, auc = evaluating(dmr, mode, test_loader, device, visual_features, text_features, outfit_item, user_idx, outfit_len)
        print("evaluating: epoch:", i, "loss:", loss, "test_auc:", auc)
        #file3.write(str(loss)+'\n')
        #file4.write(str(auc)+'\n')

    #file1.close()
    #file2.close()
    #file3.close()
    #file4.close()


if __name__ == "__main__":

    # "cpu" or "cuda:x" x is GPU index like (0,1,2,3,)
    import os
    # try:
        # os.mkdir('./model1')
    # except Exception: pass
    F(mode = 'final', hidden_dim = 512, batch_size = 256, eb_size = 128, device = 'cuda:1')
    #file = open("C:/Users/QiTianM425/Desktop/test/test.txt", 'a')
    #a = [0.1651, 0.165165, 0.16156, 0.516165]
    #np_a = np.array(a)
    #file.write('\n'.join(str(a)))
    #file.close()
    print('ffff')




