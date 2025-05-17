from torchvision import transforms
import torchvision.models as models
import argparse
import random
import numpy as np
import os
import sys
# Get the absolute path of a script
script_path = os.path.abspath(sys.argv[0])
# Get the previous directory
parent_dir = os.path.dirname(os.path.dirname(script_path))
# Add the upper level directory to sys.path
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mynet import mlp, attention, net
from data_process.Caltech import Caltechdataset
from sampler import CategoriesSampler
from subspace_projection import Subspace_Projection
from utils import pprint, set_gpu, Averager, Timer, count_acc, flip
# from modules.pairwise_coregularization import pairwise, hyperplane_pairwise
import statistics

def pairwise_trace(P,Q,Z,v):
    assert isinstance(P, torch.Tensor)
    assert isinstance(Q, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    # 获取way的值
    way = P.shape[0]
    # 初始化结果矩阵
    if v==3:
        m=3
    if v==4:
        m=6
    if v==5:
        m=10
    result = torch.zeros((m, way)).cuda()
    # 对i进行加和
    for i in range(way):
        # 计算P[i]^T P[i]
        PP = torch.mm(P[i].t(), P[i])
        # 计算Q[i]^T Q[i]
        QQ = torch.mm(Q[i].t(), Q[i])
        # 计算Z[i]^T Z[i]
        ZZ = torch.mm(Z[i].t(), Z[i])
        # 计算tr(PP QQ)和tr(PP ZZ)
        trace_PQ = torch.trace(torch.mm(PP, QQ))
        trace_PZ = torch.trace(torch.mm(PP, ZZ))
        trace_QZ = torch.trace(torch.mm(QQ, ZZ))
        # 将结果存入结果矩阵的相应位置
        result[0, i] = trace_PQ
        result[1, i] = trace_PZ
        result[2, i] = trace_QZ
    return result.T


def hadamard_sum(A, B):
    # 计算哈达玛积
    C = A * B
    # 将所有元素加和
    result = torch.sum(C)
    return result


def create_model(input_size, hidden_sizes):
    model = mlp(input_size, hidden_sizes).cuda().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.shot > 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    return model, optimizer, lr_scheduler


def create_attention(input_size):
    att = attention(input_size * args.shot, [16, 1]).cuda().float()
    optimizer = torch.optim.Adam(att.parameters(), lr=args.lr)
    if args.shot > 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    return att, optimizer, lr_scheduler


def process_view(data_shot, data_query, model, shot_num, test_way, query, lamb):
    data_mlp = model(data_shot.to(torch.float))
    data_shot = data_mlp.reshape(shot_num, test_way, -1)
    data_shot = torch.transpose(data_shot, 0, 1)
    # print(data_shot.shape)
    hyperplanes, mu = projection_pro.create_subspace(data_shot, test_way, shot_num)
    mu = mu.type_as(hyperplanes)
    label = torch.arange(test_way).repeat(query)
    label = label.type(torch.cuda.LongTensor)
    query_mlp = model(data_query.to(torch.float))
    data_query = query_mlp.type_as(hyperplanes)
    logits, discriminative_loss = projection_pro.projection_metric(data_query, hyperplanes, mu=mu)
    loss = F.cross_entropy(logits, label) + lamb * discriminative_loss
    acc = count_acc(logits, label)
    return hyperplanes, logits, loss, acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=3)
    parser.add_argument('--save-path', default=os.path.join(parent_dir, 'best_model', 'caltech'))
    parser.add_argument('--root', default=os.path.join(parent_dir, 'data_process'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lamb', type=float, default=0.03)
    parser.add_argument('--eta', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--low', action='store_true', default=False)
    parser.add_argument('--high', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--adaptive', action='store_true', default=True)

    args = parser.parse_args()
    args.subspace_dim = args.shot - 1
    pprint(vars(args))

    set_gpu(args.gpu)
    seed = args.seed


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(seed)
    if args.low:
        agg_name = 'low'
    elif args.high:
        agg_name = 'high'
    log_file_path = os.path.join(args.save_path, agg_name, '{}way-{}shot wo'.format(args.test_way, args.shot), 'acc.txt')
    sys.stdout = open(log_file_path, 'w')
    save_dir = os.path.join(args.save_path)

    testset = Caltechdataset(args)
    test_sampler = CategoriesSampler(testset.label, [16, 20], 400,
                                      args.test_way,
                                      args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=0, pin_memory=False)


    if args.low == True:
        model_view1, optimizer_view1, lr_scheduler_1 = create_model(48, [64])
        model_view2, optimizer_view2, lr_scheduler_2 = create_model(40, [64])
        model_view3, optimizer_view3, lr_scheduler_3 = create_model(254, [64])
        att1, optimizer_alpha1, lr_scheduler_alpha1 = create_attention(48 + 40)
        att2, optimizer_alpha2, lr_scheduler_alpha2 = create_attention(48 + 254)
        att3, optimizer_alpha3, lr_scheduler_alpha3 = create_attention(40 + 254)
    elif args.high == True:
        model_view1, optimizer_view1, lr_scheduler_1 = create_model(1984, [2048])
        model_view2, optimizer_view2, lr_scheduler_2 = create_model(512, [1024])
        model_view3, optimizer_view3, lr_scheduler_3 = create_model(928, [1024])
        att1, optimizer_alpha1, lr_scheduler_alpha1 = create_attention(1984 + 512)
        att2, optimizer_alpha2, lr_scheduler_alpha2 = create_attention(1984 + 928)
        att3, optimizer_alpha3, lr_scheduler_alpha3 = create_attention(512 + 928)

    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot

    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)
    print("Loading best model from validation...")
    model_view1.load_state_dict(torch.load(
        os.path.join(save_dir, agg_name, '{}way-{}shot wo'.format(args.test_way, args.shot), 'view1model_best.pth')))
    model_view2.load_state_dict(torch.load(
        os.path.join(save_dir, agg_name, '{}way-{}shot wo'.format(args.test_way, args.shot), 'view2model_best.pth')))
    model_view3.load_state_dict(torch.load(
        os.path.join(save_dir, agg_name, '{}way-{}shot wo'.format(args.test_way, args.shot), 'view3model_best.pth')))
    att1.load_state_dict(torch.load(
        os.path.join(save_dir, agg_name, '{}way-{}shot'.format(args.test_way, args.shot), 'att1_best.pth')))
    att2.load_state_dict(torch.load(
        os.path.join(save_dir, agg_name, '{}way-{}shot'.format(args.test_way, args.shot), 'att2_best.pth')))
    att3.load_state_dict(torch.load(
        os.path.join(save_dir, agg_name, '{}way-{}shot'.format(args.test_way, args.shot), 'att3_best.pth')))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['test_loss'] = []
    trlog['test_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['view1_max_acc'] = 0.0
    trlog['view2_max_acc'] = 0.0
    trlog['view3_max_acc'] = 0.0
    trlog['view1_acc_list'] = []
    trlog['view2_acc_list'] = []
    trlog['view3_acc_list'] = []
    trlog['fusion_acc_list'] = []

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        torch.cuda.empty_cache()
        model_view1.eval()
        model_view2.eval()
        model_view3.eval()

        if args.adaptive == True:
            att1.eval()
            att2.eval()
            att3.eval()


        tl = Averager()
        tl_view1 = Averager()
        tl_view2 = Averager()
        tl_view3 = Averager()

        ta = Averager()
        ta_view1 = Averager()
        ta_view2 = Averager()
        ta_view3 = Averager()


        for i, batch in enumerate(test_loader, 1):
            if args.low == True:
                data_view1, data_view2, data_view3, _, _, _, _ = [x.cuda() for x in batch]
            elif args.high == True:
                _, _, _, data_view1, data_view2, data_view3, _ = [x.cuda() for x in batch]
            p = args.shot * args.test_way
            data_shot_view1, data_query_view1 = data_view1[:p], data_view1[p:]
            data_shot_view2, data_query_view2 = data_view2[:p], data_view2[p:]
            data_shot_view3, data_query_view3 = data_view3[:p], data_view3[p:]
            #print(data_query_view1.shape)


            hyperplanes_1, logits_view1, loss_view1, acc_view1 = process_view(data_shot_view1, data_query_view1,
                                                                              model_view1, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_2, logits_view2, loss_view2, acc_view2 = process_view(data_shot_view2, data_query_view2,
                                                                              model_view2, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_3, logits_view3, loss_view3, acc_view3 = process_view(data_shot_view3, data_query_view3,
                                                                              model_view3, shot_num, args.test_way,
                                                                              args.query, args.lamb)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(logits_view1 + logits_view2 + logits_view3, label)

            tl_view1.add(loss_view1)
            tl_view2.add(loss_view2)
            tl_view3.add(loss_view3)

            ta_view1.add(acc_view1)
            ta_view2.add(acc_view2)
            ta_view3.add(acc_view3)
            ta.add(acc)


        if args.low:
            agg_name = 'low'
        elif args.high:
            agg_name = 'high'


        trlog['test_loss'].append(tl.item())
        trlog['test_acc'].append(ta.item())
        trlog['view1_acc_list'].append(ta_view1.item())
        trlog['view2_acc_list'].append(ta_view2.item())
        trlog['view3_acc_list'].append(ta_view3.item())
        trlog['fusion_acc_list'].append(ta.item())

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

        if epoch % 10 == 0:
            # print("\n=== Final Evaluation Summary ===")
            print("Epoch={}".format(epoch))
            print("view1: mean = {:.4f}, std = {:.4f}".format(
                statistics.mean(trlog['view1_acc_list']),
                statistics.stdev(trlog['view1_acc_list'])
            ))

            print("view2:  mean = {:.4f}, std = {:.4f}".format(
                statistics.mean(trlog['view2_acc_list']),
                statistics.stdev(trlog['view2_acc_list'])
            ))

            print("view3: mean = {:.4f}, std = {:.4f}".format(
                statistics.mean(trlog['view3_acc_list']),
                statistics.stdev(trlog['view3_acc_list'])
            ))
            print("MASA: mean = {:.4f}, std = {:.4f}".format(
                statistics.mean(trlog['fusion_acc_list']),
                statistics.stdev(trlog['fusion_acc_list'])
            ))
