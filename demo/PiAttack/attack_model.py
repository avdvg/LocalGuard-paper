import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

# 窃取隐私数据分两种，分类和回归。
# 分类任务

class Adv_class(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 600),
            nn.ReLU(),

            nn.Linear(600, 200),
            nn.ReLU(),

            nn.Linear(200, 100),
            nn.ReLU(),

            nn.Linear(100, target_dim)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def adversary_class_train(optim, model, feature, target):

    know_port = 0.5
    know_all = feature.shape[0]
    feature = feature[:int(know_port*know_all)]
    target = target[:int(know_port*know_all)]

    optim.zero_grad()
    pred = model(feature)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, target)
    loss.backward()
    optim.step()

    return loss.item()


def adversary_class_test(optim, model, feature, target):

    know_port = 0.2
    know_all = feature.shape[0]
    feature = feature[int(know_port*know_all):]
    target = target[int(know_port*know_all):]

    model.eval()
    pred = model(feature)
    pred = np.argmax(pred.detach().cpu(), axis=1)

    right = np.sum(np.array(pred == target.cpu()) + 0)
    acc = right/pred.shape[0]


    f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
    acc = round(acc, 3)
    try:
        auc = roc_auc_score(target.cpu(), pred.cpu(), average='micro')
        auc = round(auc, 3)
    except:
        auc = None
    f1 = round(f1, 3)

    return know_port, acc, f1
