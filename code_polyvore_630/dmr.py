import torch
import numpy as np
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import time
from model.backbones import alexnet
#from torchvision.models import alexnet


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        print(x)
        return x


class DmrFcnAttention(Module):
    def __init__(self, eb_size=128):
        super(DmrFcnAttention, self).__init__()
        self.eb_size = eb_size
        self.att = nn.Sequential(#PrintLayer(),
                                 nn.BatchNorm1d(50),
                                 #PrintLayer(),
                                 nn.Dropout(0.2),
                                 nn.Linear(self.eb_size*8, 256),
                                 #PrintLayer(),
                                 nn.BatchNorm1d(50),
                                 #PrintLayer(),
                                 nn.Sigmoid(),
                                 nn.Dropout(0.2),
                                 #PrintLayer(),
                                 nn.Linear(256, 1)

        )

    def forward(self, item_eb, item_his_eb):
        item_eb_tile = item_eb.repeat(1, item_his_eb.size()[1]) # B, 50*256
        item_eb_tile = torch.reshape(item_eb_tile, (-1, item_his_eb.size()[1], item_eb.size()[-1])) # B,50,256
        query = item_eb_tile
        dmr_all = torch.cat([query, item_his_eb, query-item_his_eb, query*item_his_eb], -1)
        atten = self.att(dmr_all)
        scores_no_softmax = torch.reshape(atten, [-1, 1, item_his_eb.size()[1]])

        scores = F.softmax(scores_no_softmax, dim=-1)

        output = torch.matmul(scores, item_his_eb)
        output = torch.sum(output, 1)

        return output, scores, scores_no_softmax


class ItemAttention(Module):
    def __init__(self, eb_size=128):
        super(ItemAttention, self).__init__()
        self.eb_size = eb_size
        self.att = nn.Sequential(nn.Linear(self.eb_size*2+16, 1),
                                 nn.Softmax(dim=-1)
                                 )

    def forward(self, his_outfit_encode, user_id):
        his_outfit = his_outfit_encode.reshape(-1, 3, self.eb_size*2)
        user_id_tile = user_id.repeat(1, 3*his_outfit_encode.size()[1]).reshape(-1, 3, 16)
        query = torch.cat([his_outfit, user_id_tile], -1)
        #print(query.size())
        atten = self.att(query).reshape([-1, 1, 3])
        score = atten.reshape(-1, his_outfit_encode.size()[1], 1, 3)

        output = torch.matmul(score, his_outfit_encode)
        return output


class GetVisual(nn.Module):
    def __init__(self, eb_size=128):
        super(GetVisual, self).__init__()
        self.eb_size = eb_size
        self.visual_nn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, self.eb_size, bias=False),
            nn.BatchNorm1d(self.eb_size)
        )

    def forward(self, x):
        x = self.visual_nn(x)
        return x


class GetText(nn.Module):
    def __init__(self, eb_size=128):
        super(GetText, self).__init__()
        self.eb_size = eb_size
        self.text_nn = nn.Sequential(
            nn.Linear(2400, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, self.eb_size, bias=False),
            nn.BatchNorm1d(self.eb_size)
        )

    def forward(self, x):
        x = self.text_nn(x)
        return x


class DMR(Module):
    def __init__(self, batch_size, eb_size=128):

        super(DMR, self).__init__()
        self.epoch = 0
        self.batch_size = batch_size
        self.eb_size = eb_size

        self.user_eb = nn.Embedding(630, 16)
        self.features = alexnet(pretrained=False)

        self.visual = GetVisual()
        self.text = GetText()

        self.dmr_fcn_attention = DmrFcnAttention(self.eb_size)
        self.item_attention = ItemAttention(self.eb_size)

        self.fc1 = nn.Sequential(
            nn.Linear(self.eb_size*2, 64)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.eb_size*2, 64)
        )

        self.build_fcn_net = nn.Sequential(
            #PrintLayer(),
            nn.BatchNorm1d(self.eb_size*6 + 65),
            #PrintLayer(),
            nn.Linear(self.eb_size*6 + 65, 256),
            #PrintLayer(),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            #PrintLayer()
        )

    def forward(self, uid, user_his, outfit):
        # pre deal
        his_v = []
        his_t = []
        for out in user_his:
            out_v, out_t = out
            out_v_feat = [self.features(x) for x in out_v]
            out_v_latent = [self.visual(x) for x in out_v_feat]
            out_t_latent = [self.text(x) for x in out_t]
            out_v_latent = torch.stack(out_v_latent, dim=1)
            out_t_latent = torch.stack(out_t_latent, dim=1)
            his_v.append(out_v_latent)
            his_t.append(out_t_latent)

        his_v_latent = torch.stack(his_v, dim=1)  # B*50*3*128
        his_t_latent = torch.stack(his_t, dim=1)
        his_latent = torch.cat([his_v_latent, his_t_latent], dim=-1)  # B*50*3*256

        user_latent = self.user_eb(uid)  # B*16

        outfit_v, outfit_t = outfit
        outfit_v_feat = [self.features(x) for x in outfit_v]
        outfit_v_latent = [self.visual(x) for x in outfit_v_feat]
        outfit_t_latent = [self.text(x) for x in outfit_t]

        outfit_v_latent = torch.stack(outfit_v_latent, dim=1)
        outfit_t_latent = torch.stack(outfit_t_latent, dim=1)
        outfit_latent = torch.cat([outfit_v_latent, outfit_t_latent], dim=-1).unsqueeze(1)  # B*1*3*256

        his_outfit_dm = self.item_attention(his_latent, user_latent)
        outfit_dm = self.item_attention(outfit_latent, user_latent)
        #print(his_outfit_dm.size())
        #print(outfit_dm.size())

        his_outfit_dm = his_outfit_dm.squeeze(2)
        outfit_dm = outfit_dm.squeeze(2).squeeze(1)
        #print(his_outfit_dm.size())
        #print(outfit_dm.size())

        his_outfit_dm_sum = torch.sum(his_outfit_dm, 1)

        att_outputs, alphas, scores_unnorm = self.dmr_fcn_attention(outfit_dm, his_outfit_dm)
        rel_i2i = torch.sum(scores_unnorm, [1, 2]).unsqueeze(-1)
        #scores = torch.sum(alphas, 1)
        # print(att_outputs)

        user_dm_eb = self.fc1(his_outfit_dm_sum)
        outfit_dm_eb = self.fc2(outfit_dm)

        inp = torch.cat([outfit_dm, att_outputs, rel_i2i, his_outfit_dm_sum, outfit_dm_eb*user_dm_eb], -1)
        # print(inp.size())
        x = self.build_fcn_net(inp)
        # print(x)
        return x
