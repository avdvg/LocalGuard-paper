import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, average_precision_score
import time
import sklearn
import argparse
import numpy as np
import random
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pickle
import sys
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from    scipy.sparse.linalg.eigen.arpack import eigsh
import math
import  pickle as pkl
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

import  numpy as np


import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
import sys

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # splits are random

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    # num_test = int(np.floor(edges.shape[0] / 60.))
    # num_val = int(np.floor(edges.shape[0] / 10.))
    num_val = int(edges.shape[0] * 0.1)
    num_test = int(edges.shape[0] * 0.6)
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # Note: these edge lists only contain sigle direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def load_ccpdata(dataset):

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/data0/BigPlatform/FL/lirongchang/localguard/datasets/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file('/data0/BigPlatform/FL/lirongchang/localguard/datasets/ind.{}.test.index'.format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer' or dataset == 'cora':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("/data0/BigPlatform/FL/lirongchang/localguard/datasets/ind.{}.{}".format(dataset, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.Tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    # print(type(ty))
    # if type(ty)== 'numpy.ndarray':
    #     print(type(ty))
    ty = torch.tensor(ty).float()
    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    # print('labels', labels)

    return adj, features, adj_train, val_edges, val_edges_false, test_edges, test_edges_false, labels

# print(load_ccpdata('cora'))
def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

# Data process for link inference training

# adj, feature, adj_train, _, _, test_edges, test_edges_false, labels = load_ccpdata('cora')
# feature = torch.tensor(feature.todense())

# adj = save_variable(adj, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj.txt')
# feature = save_variable(feature, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_feature.txt')
# adj_train = save_variable(adj_train, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_train.txt')
# test_edges = save_variable(test_edges, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_test_edges.txt')
# test_edges_false = save_variable(test_edges_false, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora__test_edges_false.txt')
# labels = save_variable(labels, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora__test_labels.txt')
# print('saving')

adj = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj.txt')
feature = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_feature.txt')
adj_train = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_train.txt')
test_edges = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_test_edges.txt')
test_edges_false = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora__test_edges_false.txt')
labels = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora__test_labels.txt')

print(labels)
print('starting')

num_nodes = adj.shape[0]
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0]) # 训练时测试的边标签。
adj_label = sparse_to_tuple(adj_label)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),  torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2])).cuda()

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight
weight_tensor = weight_tensor.to(device)

train_privacy_property = labels
test_privacy_property = labels


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    # print('adj22',adj,type(adj))
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


# adj_norm = preprocess_graph(adj_A.cpu().to_dense())  # adj_norm for party A train
# adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to(device)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split_graph(adj, feature, split_method):
    # Create new graph graph_A, graph_B
    # Function to build test set with 10% positive links
    # splits are random

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    if split_method == 'com':
        num_graph_A = int(edges.shape[0] * args.p)
        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        A_edge_idx = all_edge_idx[:num_graph_A]

        A_edges = edges[A_edge_idx]
        B_edges = np.delete(edges, A_edge_idx, axis=0)

    elif split_method == 'alone':
        num_graph_A = int(edges.shape[0] * 0.8)
        num_graph_B = int(edges.shape[0] * 0.8)
        all_edge_idx = list(range(edges.shape[0]))

        np.random.shuffle(all_edge_idx)
        A_edge_idx = all_edge_idx[:num_graph_A]
        np.random.shuffle(all_edge_idx)
        B_edge_idx = all_edge_idx[:num_graph_B]

        A_edges = edges[A_edge_idx]
        B_edges = edges[B_edge_idx]


    data_A = np.ones(A_edges.shape[0])
    data_B = np.ones(B_edges.shape[0])

    # Re-build adj matrix
    adj_A = sp.csr_matrix((data_A, (A_edges[:, 0], A_edges[:, 1])), shape=adj.shape)
    adj_A = adj_A + adj_A.T

    adj_B = sp.csr_matrix((data_B, (B_edges[:, 0], B_edges[:, 1])), shape=adj.shape)
    adj_B = adj_B + adj_B.T

    # Feature split evenly

    feature_1 = torch.split(feature, feature.size()[1] // 2, dim=1)[0]
    feature_2 = torch.split(feature, feature.size()[1] // 2, dim=1)[1]

    # feature_A = feature
    # feature_B = feature

    adj_B_tui = adj_B # For link inference attack
    adj_label_A = adj_A  # For link inference main task
    # print('333', adj_label_A)
    # adj_label_A = sparse_to_tuple(adj_label_A)
    adj_A = preprocess_adj(adj_A)
    adj_A = sparse_mx_to_torch_sparse_tensor(adj_A)

    adj_B = preprocess_adj(adj_B)
    adj_B = sparse_mx_to_torch_sparse_tensor(adj_B)

    return adj_A, adj_B, adj_B_tui, adj_label_A, feature_1, feature_2

adj_A, adj_B, _, adj_A_link, feature_1, feature_2 = split_graph(adj_train, feature, 'alone')


# adj_A = save_variable(adj_A, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_A.txt')
# adj_B = save_variable(adj_B, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_B.txt')
# adj_A_link = save_variable(adj_A_link, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_A_link.txt')
# feature_A = save_variable(feature_1, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_feature_1.txt')
# feature_B = save_variable(feature_2, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_feature_2.txt')
# print('saving2')

# adj_A 是A的邻接矩阵（训练集）， adj_A_link 是A的测试集边。
adj_A = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_A.txt')
adj_B = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_B.txt')
adj_A_link = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_adj_A_link.txt')
feature_1 = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_feature_1.txt')
feature_2 = load_variavle('/data0/BigPlatform/FL/lirongchang/localguard/tmp/cora_feature_2.txt')

adj_A_train = preprocess_graph(adj_A.cpu().to_dense())  # adj_norm for party A train
adj_A_train = torch.sparse.FloatTensor(torch.LongTensor(adj_A_train[0].T), torch.FloatTensor(adj_A_train[1]), torch.Size(adj_A_train[2])).to(device)

adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = torch.spmm(infeatn, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.dropout = 0.2

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.relu(x)
        # x = F.dropout(x,p=0.8)
        x = self.gc2(x, adj)
        # x = F.dropout(x, p=0.8)
        return x

# 构建VFL模型
#
def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred

class SplitNN(nn.Module):
    def __init__(self, models, optimizers, partition):
        super().__init__()
        self.models = models
        self.optimizers = optimizers
        self.output = [None] * (partition)

    #         self.output.to(device)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()


    def forward(self, x, stage, epoch):
        for i in range(len(x)):

            self.output[i] = self.models[i](x[i][0], x[i][1])

            if epoch == 99 and stage=='train':
                torch.save(self.output[1], 'cora_embedding_train.pth')

        # Concatenating the output of various structures in bottom part (alice's location)
        total_out = torch.cat(tuple(self.output[i] for i in range(len(self.output))), dim=1)
        second_layer_inp = total_out.detach().requires_grad_()

        self.second_layer_inp = second_layer_inp
        # pred = self.models[-1](second_layer_inp)
        pred = dot_product_decode(self.second_layer_inp)

        return pred, self.output[1]

    def backward(self):

        second_layer_inp = self.second_layer_inp
        grad = second_layer_inp.grad

        i = 0
        while i < partition - 1:
            self.output[i].backward(grad[:, hidden_sizes[1] * i: hidden_sizes[1] * (i + 1)])
            i += 1

        # This is implemented because it is not necessary that last batch is of exact same size as partitioned.
        self.output[i].backward(grad[:, hidden_sizes[1] * i:])

    def step(self):
        for opt in self.optimizers:
            opt.step()



def create_models(partition, input_size, hidden_sizes, output_size):
    models = list()
    for _ in range(1, partition):
        models.append(GCN(nfeat=716, nhid=128, nclass=hidden_sizes[1]).cuda())

    models.append(GCN(nfeat=716, nhid=128, nclass=hidden_sizes[1]).cuda())

    models.append(nn.Sequential(nn.Linear(hidden_sizes[1] * partition, hidden_sizes[2]),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(hidden_sizes[2], output_size),
                                nn.LogSoftmax(dim=1)
                                ).cuda())
    return models

input_size = 7
hidden_sizes = [128, 128, 64]
output_size = 3

partition = 2
models = create_models(partition, input_size, hidden_sizes, output_size)

optimizers = [optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) for model in
              models]

optimizers = [optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) for model in
              models]

splitNN = SplitNN(models, optimizers, partition).cuda()
#
# # 训练模型
#
def masked_loss(out, label, mask):
    # print(out)
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss

def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def train(epoch, x, splitnn):
    splitnn.zero_grads()
    pred, total_ = splitnn.forward(x, 'train', epoch)

    loss = F.binary_cross_entropy(pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    # print('d',loss.item())
    loss.backward()
    splitnn.backward()
    splitnn.step()
    return loss.item()
#
#
def test_cda(epoch, x, splitnn):
    splitnn.eval()
    pred, total_ = splitnn.forward(x, 'test',epoch)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, pred.cpu())

    return test_roc, test_ap

def adversary():

    adv_train = torch.load('cora_embedding_train.pth').cuda()
    adv_test = torch.load('cora_embedding_train.pth').cuda()

    from PiAttack.attack_model import Adv_class
    from PiAttack.attack_model import adversary_class_train
    from PiAttack.attack_model import adversary_class_test

    AttackModel = Adv_class(latent_dim=128, target_dim=7).cuda()
    optim_ = optim.Adam(AttackModel.parameters(), lr=0.01, weight_decay=1e-8)
    print(train_privacy_property.shape)
    for i in range(100):

        loss = adversary_class_train(optim_, AttackModel, adv_train, train_privacy_property.cuda())
        # print('the attack train epoch {} and loss is {}'.format(i, loss))
    know_port, acc_0, f1_0 = adversary_class_test(optim_, AttackModel, adv_test, test_privacy_property.cuda())
    print('the attack test epoch {}: the knowledge {} ==> acc is {}, f1 is {}.'.format(i, know_port, acc_0, f1_0))




epochs = 100
loss_list = list()
cda_list = list()


adj = torch.tensor(adj.getnnz(), dtype=torch.float32).cuda()
feature_1 = torch.tensor(feature_1, dtype=torch.float32).cuda()
feature_2 = torch.tensor(feature_2, dtype=torch.float32).cuda()


print('starting!')
for epoch in range(epochs):
    total_loss = 0

    loss = train(epoch, [(feature_1.cuda(), adj_A.cuda()), (feature_2.cuda(), adj_B.cuda())], splitNN)

    loss_list.append(loss)
    print(f"Epoch: {epoch+1}... Training Loss: {loss}")

    test_roc, test_ap = test_cda(epoch, [(feature_1, adj_A.cuda()), (feature_2, adj_B.cuda())], splitNN)
    # cda, precision, recall, f1 = torch.round(cda, 3), torch.round(precision,3), torch.round(recall, 3), torch.round(f1, 3)
    cda_list.append(test_roc)

    print(f"Epoch: {epoch + 1}... testing auc is: {test_roc} precition {test_ap}")


print('the max cda in the exp is', max(cda_list))


adversary()
