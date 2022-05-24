# 生成LocalGuard噪声，优化目标分为2范数约束以及嵌入特征与随机分布隐私标签对应

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Optim_noise(nn.Module):
    def __init__(self, x_dim, y_dim):  # feature0 = 3, feature1 = 6
        super(Optim_noise, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.x_dim, 300),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 200),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(200, self.y_dim),
            nn.ReLU(),

            )

    def forward(self, X):
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        out = self.mlp(X)

        return out

from torch.autograd import Variable

def LocalGuard(emb, ee, st,p):


    if (ee % 1 == 0) and st=='train':
        raw_emb = emb.clone()
        noise_num = emb.shape[0] * emb.shape[1]
        # raw_emb = Variable(raw_emb, requires_grad=True)

        model = Optim_noise(noise_num, noise_num).cuda()
        init_input = torch.ones(noise_num).float().cuda()
        print(init_input.shape)
        # print(emb.shape[0]*emb.shape[1])

        # emb = Variable(emb, requires_grad=True)

        optimizer = torch.optim.Adam([
            dict(params=model.parameters()),
        ], lr=1, weight_decay=1e-5)

        import random

        lam = 1
        a = []
        leibie = 7
        for j in range(emb.shape[0] // leibie):
            for i in range(leibie):
                a.append(i)
        # print(len(a))
        while len(a) < emb.shape[0]:
            a.append(0)
        # print(a.shape)

        random.shuffle(a)
        print(a)

        c = torch.tensor(a).reshape(emb.shape[0]).cuda()


        for e in range(50):
            optimizer.zero_grad()
            model.train()

            injection_noise = model(init_input).reshape(emb.shape)

            new_emb = injection_noise + emb  # 0.5*F.l1_loss(new_emb,raw_emb) +F.l1_loss(new_emb,raw_emb) +

            # loss1 = nn.MSELoss()(new_emb,raw_emb) +lam*nn.CrossEntropyLoss()(new_emb,c) #nn.KLDivLoss()(new_emb,raw_emb)
            # loss2 = nn.MSELoss()(new_emb, raw_emb)
            loss2 = nn.L1Loss()(new_emb, raw_emb)
            loss3 = nn.CrossEntropyLoss()(new_emb, c)
            # loss3 = F.nll_loss(nn.Sigmoid()(new_emb),c)
            loss1 = 1*lam * loss3# +0.00001 * loss2
            loss1.backward(retain_graph=True)
            # print([x.grad for x in optimizer.param_groups[0]['params']])
            optimizer.step()
            print(loss1.item())

        np.save('./noise.npy', injection_noise.cpu().detach().numpy())

        predd = np.argmax(new_emb.detach().cpu(), axis=1)
        right = np.sum(np.array(predd == c.cpu()) + 0)
        #
        print('infer acc', right / predd.shape[0])

    else:
        injection_noise = torch.from_numpy(np.load('./noise.npy')).cuda()


    return injection_noise










