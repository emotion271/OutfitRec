import torch

from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from  torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.init import uniform_


class DmrFcnAttention(Module):
    def __init__(self, eb_size=128):
        super(DmrFcnAttention, self).__init__()
        self.eb_size = eb_size
        self.fc1 = nn.Linear(self.eb_size*2, self.eb_size*2)
        self.prelu = nn.PReLU()
        self.att = nn.Sequential(nn.Linear(self.eb_size*8,512),#80
                                 nn.Sigmoid(),
                                 nn.Linear(512, 40),
                                 nn.Sigmoid(),
                                 nn.Linear(40, 1)

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
        query = self.fc1(query)
        query = self.prelu(query)
        dmr_all = cat([query, item_his_eb, query-item_his_eb, query*item_his_eb], -1)
        atten = self.att(dmr_all)
        atten = torch.reshape(atten, [-1, 1, item_his_eb.size()[1]])
        scores = atten

        key_masks = mask.unsqueeze(1)
        paddings = torch.ones_like(scores) * (-2**32 + 1)
        paddings_no_softmax = torch.zeros_like(scores)
        scores = torch.where(key_masks, scores, paddings)
        scores_no_softmax = torch.where(key_masks, scores, paddings_no_softmax)

        scores = F.softmax(scores, dim=1)

        output = torch.matmul(scores, item_his_eb)
        output = torch.sum(output, 1)

        return output, scores, scores_no_softmax


class ItemAttention(Module):
    def __init__(self, eb_size=128):
        super(ItemAttention, self).__init__()
        self.eb_size = eb_size
        self.att = nn.Sequential(nn.Linear(self.eb_size*2+32, 64),
                                 nn.Sigmoid(),
                                 nn.Linear(64, 16),
                                 nn.Sigmoid(),
                                 nn.Linear(16, 1),
                                 nn.Softmax(dim=1))

    def forward(self, his_outfit_encode, user_id):
        his_outfit = his_outfit_encode.reshape(-1, 20, self.eb_size*2)#B*50,20,d*2
        #print(his_outfit.size())
        user_id_tile = user_id.repeat(1, 20*his_outfit_encode.size()[1]).reshape(-1, 20, 32)#B*50,20,32
        #print(user_id_tile.size())
        query = cat([his_outfit, user_id_tile], -1)
        atten = self.att(query)
        atten = torch.reshape(atten, [-1, 1, 20])
        score = atten.reshape(-1, his_outfit_encode.size()[1], 1, 20)
        output = torch.matmul(score, his_outfit_encode)
        #print(output.size())

        return output

class ItemAttention_nouser(Module):
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


class DMR(Module):
    def __init__(self, batch_size,
                 visual_feature_dim=2048, hidden_dim=512,
                 eb_size=128,
                 uniform_value=0.5):

        super(DMR, self).__init__()
        self.epoch = 0
        self.batch_size = batch_size
        self.eb_size = eb_size
        self.uniform_value = uniform_value
        self.hidden_dim = hidden_dim
        # print(list(self.features.keys()))
        # self.user_eb = nn.Embedding(3570, 32)
        self.visual_nn = Sequential(
            Linear(visual_feature_dim, self.hidden_dim),
            Sigmoid(),
            Linear(self.hidden_dim, self.eb_size),
            Sigmoid()
        )

        self.text_nn = Sequential(
            Linear(300, self.eb_size),
            Sigmoid()
        )

        self.item_attention = ItemAttention_nouser(self.eb_size)
        # print('Module already prepared, {} users, {} items'.format(len(user_set), len(item_set)))

        self.fc1 = nn.Sequential(
            nn.Linear(self.eb_size*2, 64),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.eb_size*2, 64),
            nn.Sigmoid()
        )

        self.build_fcn_net = nn.Sequential(
            nn.BatchNorm1d(self.eb_size*6 + 65),
            nn.Linear(self.eb_size*6 + 65, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, his_outfit_text, his_outfit_visual, mask, outfit_text, outfit_visual, uid, user_idx, **args):
        # pre deal
      

        #print(uid)
        user_id = cat([self.user_eb(user_idx[str(key.item())]) for key in uid], 0)
        #print(user_id)
        user_id = user_id.reshape(-1, 32)
        #print(user_id)
        #print(his_outfit_text.size())
        his_outfit_text.reshape(-1, 300)
        his_outfit_text_dm = self.text_nn(his_outfit_text).reshape(-1, 50, 20, self.eb_size)
        his_outfit_visual.reshape(-1, 2048)
        his_outfit_visual_dm = self.visual_nn(his_outfit_visual).reshape(-1, 50, 20, self.eb_size)
        outfit_text.reshape(-1, 300)
        outfit_text_dm = self.text_nn(outfit_text).reshape(-1, 1, 20, self.eb_size)
        outfit_visual.reshape(-1, 2048)
        outfit_visual_dm = self.visual_nn(outfit_visual).reshape(-1, 1, 20, self.eb_size)


        #print(his_outfit_text_dm.size())
        #print(his_outfit_visual_dm.size())
        #print(outfit_text_dm.size())
        #print(outfit_visual_dm.size())

        his_outfit_encode = cat([his_outfit_text_dm, his_outfit_visual_dm], -1)
        outfit_encode = cat([outfit_text_dm, outfit_visual_dm], -1)

        #print(his_outfit_encode.size())
        #print(outfit_encode.size())

        his_outfit_dm = self.item_attention(his_outfit_encode, user_id)
        outfit_dm = self.item_attention(outfit_encode, user_id)
        #his_outfit_dm = self.item_attention(his_outfit_encode)
        #outfit_dm = self.item_attention(outfit_encode)

        his_outfit_dm = his_outfit_dm.squeeze(2)
        outfit_dm = outfit_dm.squeeze(2).squeeze(1)


        his_outfit_dm_sum = torch.sum(his_outfit_dm, 1)

        att_outputs, alphas, scores_unnorm = self.dmr_fcn_attention(outfit_dm, his_outfit_dm, mask)
        rel_i2i = torch.sum(scores_unnorm, [1, 2]).unsqueeze(-1)
        scores = torch.sum(alphas, 1)
        # print(att_outputs)

        user_dm_eb = self.fc1(his_outfit_dm_sum)
        outfit_dm_eb = self.fc2(outfit_dm)

        inp = cat([outfit_dm, att_outputs, rel_i2i, his_outfit_dm_sum, outfit_dm_eb*user_dm_eb], -1)
        # print(inp.size())
        x = self.build_fcn_net(inp)
        # print(x)
        return x


        # part one General


