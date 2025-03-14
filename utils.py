import os
import shutil
import time
import pprint
import argparse

import torch
import torch.nn as nn
import torch.autograd.variable as Variable

from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
import numpy as np
from collections import OrderedDict

class GaussianNoise(nn.Module):

    def __init__(self, batch_size, input_shape=(3, 84, 84), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std

    def forward(self, x, std=0.15):
        noise = Variable(torch.zeros(x.shape).cuda())
        noise = noise.data.normal_(0, std=std)
        return x + noise


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def clone(tensor):
    """Detach and clone a tensor including the ``requires_grad`` attribute.

    Arguments:
        tensor (torch.Tensor): tensor to clone.
    """
    cloned = tensor.clone()#tensor.detach().clone()
    # cloned.requires_grad = tensor.requires_grad
    # if tensor.grad is not None:
    #     cloned.grad = clone(tensor.grad)
    return cloned

def clone_state_dict(state_dict):
    """Clone a state_dict. If state_dict is from a ``torch.nn.Module``, use ``keep_vars=True``.

    Arguments:
        state_dict (OrderedDict): the state_dict to clone. Assumes state_dict is not detached from model state.
    """
    return OrderedDict([(name, clone(param)) for name, param in state_dict.items()])

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def prediction(logits):
    pred = torch.argmax(logits, dim=1)
    return pred

def confusion_matrix(logits, labels, conf_matrix):
    preds = torch.argmax(logits, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def d_mink_acc(logits, label, way, query):
    device = torch.device('cuda:0')
    values, indicies = logits.topk(6, dim=1)
    pred = torch.zeros(way * query).to(device)
    # print(f'希望和label一样：{pred.shape}')
    # print(label.shape)
    row = indicies.shape[0]
    for i in range(row):
        score = torch.zeros(way).to(device)
        d_max5 = indicies[i]
        # print(f'索引：{d_max5}')
        c = 0.
        for x in d_max5:
            for y in range(way):
                if x % way == y:
                    score[y] = score[y]+1-c
            c = c + 0.1
        # print(f'分数：{score}')
        predict = torch.argmax(score, dim=0)
        pred[i] = predict
    pred.to(device)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    #logits = -((a - b)**2).sum(dim=2)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

def set_protocol(data_path, protocol, test_protocol, subset=None):
    train = []
    val = []

    all_set = ['shn', 'hon', 'clv', 'clk', 'gls', 'scl', 'sci', 'nat', 'shx', 'rel']

    if subset is not None:
        train.append(data_path + '/crops_' + subset + '/')
        val.append(data_path + '/crops_' + subset + '/')

    if protocol == 'p1':
        for i in range(3):
            train.append(data_path + '/crops_' + all_set[i])
    elif protocol == 'p2':
        for i in range(3, 6):
            train.append(data_path + '/crops_' + all_set[i])
    elif protocol == 'p3':
        for i in range(6, 8):
            train.append(data_path + '/crops_' + all_set[i])
    elif protocol == 'p4':
        for i in range(8, 10):
            train.append(data_path + '/crops_' + all_set[i])

    if test_protocol == 'p1':
        for i in range(3):
            val.append(data_path + '/crops_' + all_set[i])
    elif test_protocol == 'p2':
        for i in range(3, 6):
            val.append(data_path + '/crops_' + all_set[i])
    elif test_protocol == 'p3':
        for i in range(6, 8):
            val.append(data_path + '/crops_' + all_set[i])
    elif test_protocol == 'p4':
        for i in range(8, 10):
            val.append(data_path + '/crops_' + all_set[i])


    return train, val




def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]


def perturb(data):

    randno = np.random.randint(0, 5)
    if randno == 1:
        return torch.cat((data, data.flip(3)), dim=0)
    elif randno == 2: #180
        return torch.cat((data, data.flip(2)), dim=0)
    elif randno == 3: #90
        return torch.cat((data, data.transpose(2,3)), dim=0)
    else:
        return torch.cat((data, data.transpose(2, 3).flip(3)), dim=0)



def pinjie(x, shot, way):
    y = torch.zeros(shot*way, 3000)
    for i in range(shot*way):
        y[i] = torch.cat((x[i+0], x[shot*way+i], x[shot*way*2+i]), dim=-1)
    return y



def pairwise_coregularization(hyperplanes_1, hyperplanes_2, hyperplanes_3, lamb1, lamb2, lamb3, train_way):
    p_1 = hyperplanes_1.squeeze()  # front
    p_2 = hyperplanes_2.squeeze()  # side
    p_3 = hyperplanes_3.squeeze()  # top  #[3*1024]

    if len(p_1.size()) == 3:
        p_1 = p_1.view(train_way, -1)
        p_2 = p_2.view(train_way, -1)
        p_3 = p_3.view(train_way, -1)
        loss_lambda = lamb1 * ((p_1.mm(p_1.T.mm(p_2.mm(p_2.T)))).trace())
        +lamb2 * ((p_1.mm(p_1.T.mm(p_3.mm(p_3.T)))).trace())
        +lamb3 * ((p_2.mm(p_2.T.mm(p_3.mm(p_3.T)))).trace())
    else:
        loss_lambda = lamb1 * ((p_1.mm(p_1.T.mm(p_2.mm(p_2.T)))).trace())
        +lamb2 * ((p_1.mm(p_1.T.mm(p_3.mm(p_3.T)))).trace())
        +lamb3 * ((p_2.mm(p_2.T.mm(p_3.mm(p_3.T)))).trace())

    return loss_lambda



def centroid_coregularization(hyperplanes_1, hyperplanes_2, hyperplanes_3, hyperplanes_co, subspace_dim, lamb1, lamb2, lamb3):
    loss_lambda = 0.0
    if subspace_dim == 1:
        p_1 = hyperplanes_1.squeeze()  # front
        p_2 = hyperplanes_2.squeeze()  # side
        p_3 = hyperplanes_3.squeeze()  # top  #[3*1024]
        p_co = hyperplanes_co.squeeze()
        loss_lambda = lamb1 * (p_1.matmul(p_1.transpose(1, 2).matmul(p_co.matmul(p_co.transpose(1, 2)))).trace())
        +lamb2 * ((p_2.matmul(p_2.transpose(1, 2).matmul(p_co.matmul(p_co.transpose(1, 2))))).trace())
        +lamb3 * ((p_3.matmul(p_3.transpose(1, 2).matmul(p_co.matmul(p_co.transpose(1, 2))))).trace())
    else:
        p_1 = hyperplanes_1.squeeze().T  # front
        p_2 = hyperplanes_2.squeeze().T  # side
        p_3 = hyperplanes_3.squeeze().T  # top  #[3*1024*subspace_dim]
        p_co = hyperplanes_co.squeeze().T
        for i in range(subspace_dim):
            loss_lambda = lamb1 * (p_1[i].matmul(p_1[i].T.matmul(p_co[i].matmul(p_co[i].T))).trace())
            +lamb2 * ((p_2[i].matmul(p_2[i].T.matmul(p_co[i].matmul(p_co[i].T)))).trace())
            +lamb3 * ((p_3[i].matmul(p_3[i].T.matmul(p_co[i].matmul(p_co[i].T)))).trace())
            loss_lambda = loss_lambda + loss_lambda
    return loss_lambda



def shared_loss(hyperplanes_1, hyperplanes_2, hyperplanes_3, hyperplanes_co, subspace_dim):
    loss_shared = 0.0
    # loss_shared = -torch.norm(hyperplanes_1-hyperplanes_co, p=2)- \
    #               torch.norm(hyperplanes_2-hyperplanes_co, p=2)- \
    #               torch.norm(hyperplanes_3-hyperplanes_co, p=2)
    if subspace_dim == 1:
        p_1 = hyperplanes_1.squeeze()  # front
        p_2 = hyperplanes_2.squeeze()  # side
        p_3 = hyperplanes_3.squeeze()  # top  #[3*1024]
        p_co = hyperplanes_co.squeeze()
        loss_shared = (p_1.matmul(p_1.transpose(1, 2).matmul(p_co.matmul(p_co.transpose(1, 2)))).trace()) + \
                      ((p_2.matmul(p_2.transpose(1, 2).matmul(p_co.matmul(p_co.transpose(1, 2))))).trace()) + \
                      ((p_3.matmul(p_3.transpose(1, 2).matmul(p_co.matmul(p_co.transpose(1, 2))))).trace())
    else:
        p_1 = hyperplanes_1.squeeze().T  # front
        p_2 = hyperplanes_2.squeeze().T  # side
        p_3 = hyperplanes_3.squeeze().T  # top  #[3*1024*subspace_dim]
        p_co = hyperplanes_co.squeeze().T
        for i in range(subspace_dim):
            loss_lambda = (p_1[i].matmul(p_1[i].T.matmul(p_co[i].matmul(p_co[i].T))).trace()) + \
                          ((p_2[i].matmul(p_2[i].T.matmul(p_co[i].matmul(p_co[i].T)))).trace()) + \
                          ((p_3[i].matmul(p_3[i].T.matmul(p_co[i].matmul(p_co[i].T)))).trace())
            loss_shared = loss_shared + loss_shared

    loss_shared = (hyperplanes_1.matmul((torch.transpose(hyperplanes_1, 1, 2)).matmul(
        hyperplanes_co.matmul(torch.transpose(hyperplanes_co, 1, 2)))).trace()) + \
                  (hyperplanes_2.matmul((torch.transpose(hyperplanes_2, 1, 2)).matmul(
                      hyperplanes_co.matmul(torch.transpose(hyperplanes_co, 1, 2)))).trace()) + \
                  (hyperplanes_3.matmul((torch.transpose(hyperplanes_3, 1, 2)).matmul(
                      hyperplanes_co.matmul(torch.transpose(hyperplanes_co, 1, 2)))).trace())
    return loss_shared


def contrast_loss(hyperplanes_1, train_way, query, subspace_dim, T, query_view1):
    contrast_loss = 0.0

    sim = torch.zeros(train_way, train_way*query)
    for c in range(train_way):
        for i in range(train_way * query):
            similar = torch.cosine_similarity(hyperplanes_1[c].squeeze().T, query_view1[i].unsqueeze(0))*T
            similar = torch.exp(similar)
            sim[c, i] = torch.sum(similar)
            i = i+1
        c = c+1

    log = 0.0
    for c in range(train_way):
        log = log + torch.log(torch.div(torch.sum(sim[c, c*query: (c+1)*query]), torch.sum(sim[c])))
        c = c+1

    contrast_loss = torch.div(-log, train_way)

    return  contrast_loss




