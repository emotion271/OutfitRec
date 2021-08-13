import torch
import numpy as np

from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from  torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.init import uniform_


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        print(x)
        return x


class DmrFcnAttention(nn.Module):
    def __init__(self, eb_size=128):
        super(DmrFcnAttention, self).__init__()
        self.eb_size = eb_size
        #self.fc1 = nn.Linear(self.eb_size*2+8, self.eb_size*2+8)
        #self.prelu = nn.PReLU()
        self.att = nn.Sequential(nn.BatchNorm1d(50),
                                 nn.Dropout(0.2),
                                 nn.Linear(self.eb_size*8+32, 256),#80
                                 nn.BatchNorm1d(50),
                                 nn.Sigmoid(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 40),
                                 nn.BatchNorm1d(50),
                                 nn.Sigmoid(),
                                 nn.Linear(40, 1),
                                 nn.BatchNorm1d(50),
                                 nn.Sigmoid()

        )

    def forward(self, item_eb, item_his_eb, mask):
        #print(mask)
        mask = torch.eq(mask, torch.ones_like(mask))
        #print("dmr_____________________fcn")
        #print(item_eb)
        #print(item_eb.size())
        #print(mask.size())
        item_eb_tile = item_eb.repeat(1, mask.size()[1]) # B, 50*256
        #print(item_eb_tile.size())
        item_eb_tile = torch.reshape(item_eb_tile, (-1, mask.size()[1], item_eb.size()[-1])) # B,50,256
        #print(item_eb_tile.size())
        query = item_eb_tile
        #print(query.size())
        #query = self.fc1(query)
        #query = self.prelu(query)
        dmr_all = cat([query, item_his_eb, query-item_his_eb, query*item_his_eb], -1)
        #print(dmr_all.size())
        atten = self.att(dmr_all)
        atten = torch.reshape(atten, [-1, 1, item_his_eb.size()[1]])
        scores = atten

        key_masks = mask.unsqueeze(1)
        paddings = torch.ones_like(scores) * (-2**32 + 1)
        paddings_no_softmax = torch.zeros_like(scores)
        scores = torch.where(key_masks, scores, paddings)
        scores_no_softmax = torch.where(key_masks, scores, paddings_no_softmax)

        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, item_his_eb)
        output = torch.sum(output, 1)

        return output, scores, scores_no_softmax


class ItemAttention(nn.Module):
    def __init__(self, eb_size=128):
        super(ItemAttention, self).__init__()
        self.eb_size = eb_size
        self.att = nn.Sequential(#nn.BatchNorm1d(20),
                                 nn.Linear(self.eb_size*2+32+8, 32),
                                 #nn.BatchNorm1d(20),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 1),
                                 #nn.BatchNorm1d(20),
                                 nn.Sigmoid())
        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, his_outfit_encode, user_id, outfit_mask):
        his_outfit = his_outfit_encode.reshape(-1, 20, self.eb_size*2+8)
        #print("111111111111111111")
        #print(his_outfit.size())
        user_id_tile = user_id.repeat(1, 20*his_outfit_encode.size()[1]).reshape(-1, 20, 32)
        #print(user_id_tile.size())
        #print(user_id_tile.size())
        query = cat([his_outfit, user_id_tile], -1)
        #print(query.size())
        atten = self.att(query)
        atten = torch.reshape(atten, [-1, 1, 20])
        score = atten.reshape(-1, his_outfit_encode.size()[1], 1, 20)

        key_masks = outfit_mask.unsqueeze(2)
        #print('key_masks', key_masks)
        paddings = torch.ones_like(score) * (-2 ** 32 + 1)
        scores = torch.where(key_masks > 0, score, paddings)
        #print('scores', scores)
        score_norm = nn.Softmax(dim=-1)(scores)
        score_norm = self.attn_dropout(score_norm)
        #print('score_norm', score_norm)
        #weight = score_norm.data.cpu().numpy()
        #if weight.shape[1] == 1:
            #np.save('cate_weight1.npy', weight)
        #else:
            #np.save('cate_weight.npy',weight)
        #print(score.size())
        output = torch.matmul(score_norm, his_outfit_encode)
        #feat = output.data.cpu().numpy()
        #if feat.shape[1] == 50:
            #np.save('feat_norm.npy', feat)
        #print(output.size())
        #assert (0>1)

        return output


class ItemAttention_nouser(nn.Module):
    def __init__(self, eb_size=128):
        super(ItemAttention_nouser, self).__init__()
        self.eb_size = eb_size
        self.att = nn.Sequential(nn.Linear(self.eb_size*2, 32),
                                nn.Sigmoid(),
                                nn.Linear(32, 1),
                                nn.Softmax(dim=1))

    def forward(self, his_outfit_encode):
        query = his_outfit_encode.reshape(-1,20,self.eb_size*2)
        atten = self.att(query)
        atten = torch.reshape(atten, [-1, 1, 20])
        score = atten.reshape(-1, his_outfit_encode.size()[1], 1, 20)
        output = torch.matmul(score, his_outfit_encode)

        return output


class GetVisual(nn.Module):
    def __init__(self, eb_size=128):
        super(GetVisual, self).__init__()
        self.eb_size = eb_size
        self.visual_nn = nn.Sequential(
            # PrintLayer(),
            # nn.BatchNorm1d(2048),
            nn.Dropout(),
            # PrintLayer(),
            nn.Linear(2048, 512),
            # PrintLayer(),
            # nn.BatchNorm1d(512),
            # PrintLayer(),
            nn.ReLU(),
            # PrintLayer(),
            nn.Dropout(),
            # PrintLayer(),
            nn.Linear(512, self.eb_size, bias=False),
            # PrintLayer()
            nn.BatchNorm1d(self.eb_size)
            # PrintLayer(),
            # nn.Sigmoid(),
            # PrintLayer()
        )

    def forward(self, x, outfit_mask):
        l = x.size()[1]
        x = x.reshape(-1, 2048)
        x = self.visual_nn(x).reshape(-1, l, 20, self.eb_size)
        outfit_mask = outfit_mask.unsqueeze(-1)
        out = x * outfit_mask
        return out


class GetText(nn.Module):
    def __init__(self, eb_size=128):
        super(GetText, self).__init__()
        self.eb_size = eb_size
        self.text_nn = nn.Sequential(
            # nn.BatchNorm1d(300),
            nn.Linear(300, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.eb_size, bias=False),
            nn.BatchNorm1d(self.eb_size)
            # nn.Sigmoid()
        )

    def forward(self, x, outfit_mask):
        l = x.size()[1]
        x = x.reshape(-1, 300)
        x = self.text_nn(x).reshape(-1, l, 20, self.eb_size)
        outfit_mask = outfit_mask.unsqueeze(-1)
        out = x * outfit_mask
        return out


class DMR(nn.Module):
    def __init__(self, batch_size,
                 visual_feature_dim=2048,
                 eb_size=128):

        super(DMR, self).__init__()
        self.epoch = 0
        self.batch_size = batch_size
        self.eb_size = eb_size

        self.user_eb = nn.Embedding(3570, 32)
        self.cate_eb = nn.Embedding(46, 8)

        self.visual = GetVisual()
        self.text = GetText()

        self.dmr_fcn_attention = DmrFcnAttention(self.eb_size)
        self.item_attention = ItemAttention(self.eb_size)

        self.fc1 = nn.Sequential(
            nn.Linear(self.eb_size*2+8, 64),
            #nn.BatchNorm1d(64),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.eb_size*2+8, 64),
            #nn.BatchNorm1d(64),
            nn.Sigmoid()
        )

        self.build_fcn_net = nn.Sequential(
            # PrintLayer(),
            nn.BatchNorm1d(self.eb_size * 6 + 65 + 24),
            # PrintLayer(),
            nn.Linear(self.eb_size * 6 + 65 + 24, 256),
            # PrintLayer(),
            nn.PReLU(),
            nn.Linear(256, 128),
            # PrintLayer(),
            nn.PReLU(),
            nn.Linear(128, 1),
            # PrintLayer(),
            nn.Sigmoid()
            # PrintLayer()
        )

    def forward(self, his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, user_idx, outfit_his_mask, outfit_mask, his_outfit_cate, outfit_cate):
        # pre deal
        # print(source.size())

        #print(his_outfit_cate.size())
        #print(outfit_cate.size())
        #print(outfit_his_mask.size())
        #print(outfit_mask.size())

        his_outfit_cate.reshape(-1, 1)
        his_outfit_cate_dm = self.cate_eb(his_outfit_cate).reshape(-1, 50, 20, 8)

        outfit_cate.reshape(-1, 1)
        outfit_cate_dm = self.cate_eb(outfit_cate).reshape(-1, 1, 20, 8)

        #print(uid)
        user_id = cat([self.user_eb(user_idx[str(key.item())]) for key in uid], 0)
        #print(user_id.size())
        #print(user_id)
        user_id = user_id.reshape(-1, 32)
        #print(user_id.size())
        #print(user_id)
        #print(his_outfit_text.size())

        '''
        his_outfit_text.reshape(-1, 300)
        his_outfit_text_dm = self.text_nn(his_outfit_text).reshape(-1, 50, 20, self.eb_size)
        his_outfit_visual.reshape(-1, 2048)
        his_outfit_visual_dm = self.visual_nn(his_outfit_visual).reshape(-1, 50, 20, self.eb_size)
        outfit_text.reshape(-1, 300)
        outfit_text_dm = self.text_nn(outfit_text).reshape(-1, 1, 20, self.eb_size)
        outfit_visual.reshape(-1, 2048)
        outfit_visual_dm = self.visual_nn(outfit_visual).reshape(-1, 1, 20, self.eb_size)
        '''

        his_outfit_text_dm = self.text(his_outfit_text, outfit_his_mask)
        his_outfit_visual_dm = self.visual(his_outfit_visual, outfit_his_mask)
        outfit_text_dm = self.text(outfit_text, outfit_mask)
        outfit_visual_dm = self.visual(outfit_visual, outfit_mask)

        his_outfit_encode = cat([his_outfit_text_dm, his_outfit_visual_dm, his_outfit_cate_dm], -1)
        outfit_encode = cat([outfit_text_dm, outfit_visual_dm, outfit_cate_dm], -1)
        #print(his_outfit_encode.size())
        #print(outfit_encode.size())

        his_outfit_dm = self.item_attention(his_outfit_encode, user_id, outfit_his_mask)
        outfit_dm = self.item_attention(outfit_encode, user_id, outfit_mask)
        #print(his_outfit_dm)
        #print(outfit_dm)
        #his_outfit_dm = self.item_attention(his_outfit_encode)
        #outfit_dm = self.item_attention(outfit_encode)

        his_outfit_dm = his_outfit_dm.squeeze(2)
        outfit_dm = outfit_dm.squeeze(2).squeeze(1)

        his_outfit_dm_sum = torch.sum(his_outfit_dm, 1)

        att_outputs, alphas, scores_unnorm = self.dmr_fcn_attention(outfit_dm, his_outfit_dm, mask)
        rel_i2i = torch.sum(scores_unnorm, [1, 2]).unsqueeze(-1)
        #scores = torch.sum(alphas, 1)
        # print(att_outputs)

        user_dm_eb = self.fc1(his_outfit_dm_sum)
        outfit_dm_eb = self.fc2(outfit_dm)

        inp = cat([outfit_dm, att_outputs, rel_i2i, his_outfit_dm_sum, outfit_dm_eb*user_dm_eb], -1)
        # print(inp.size())
        x = self.build_fcn_net(inp)
        # print(x)
        return x


        # part one General


