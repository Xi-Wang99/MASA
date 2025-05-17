from torchvision import transforms
import torchvision.models as models
import argparse
import os.path as osp
import os
import random
import numpy as np
import sys
# sys.path.append('/home/wangxi/dsn_multiview')

script_path = os.path.abspath(sys.argv[0])
# Get the previous directory
parent_dir = os.path.dirname(os.path.dirname(script_path))
# Add the upper level directory to sys.path
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mynet import mlp, attention
from data_process.NUS_WIDE_OBJECT import NUS_WIDE_OBJECTdataset
from sampler import CategoriesSampler
from subspace_projection import Subspace_Projection
from utils import pprint, set_gpu, Averager, Timer, count_acc, flip
import time
import statistics


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

def process_view(data_shot, data_query, model, shot_num, train_way, query, lamb):
    data_mlp = model(data_shot.to(torch.float))
    data_shot = data_mlp.reshape(shot_num, train_way, -1)
    data_shot = torch.transpose(data_shot, 0, 1)
    hyperplanes, mu = projection_pro.create_subspace(data_shot, train_way, shot_num)
    mu = mu.type_as(hyperplanes)
    label = torch.arange(train_way).repeat(query)
    label = label.type(torch.cuda.LongTensor)
    query_mlp = model(data_query.to(torch.float))
    data_query = query_mlp.type_as(hyperplanes)
    logits, discriminative_loss = projection_pro.projection_metric(data_query, hyperplanes, mu=mu)
    loss = F.cross_entropy(logits, label) + lamb * discriminative_loss
    acc = count_acc(logits, label)
    return hyperplanes, logits, loss, acc


def pairwise_5view(A, B, C, D, E):
    assert all(isinstance(matrix, torch.Tensor) for matrix in [A, B, C, D, E])
    way = A.shape[0]
    result = torch.zeros((10, way)).cuda()
    for i in range(way):
        row = 0
        for j, matrix1 in enumerate([A, B, C, D, E]):
            for matrix2 in [A, B, C, D, E][j+1:]:
                # matrix1[i]^T matrix1[i]
                P = torch.mm(matrix1[i].t(), matrix1[i])
                # matrix2[i]^T matrix2[i]
                Q = torch.mm(matrix2[i].t(), matrix2[i])
                # tr(P Q)
                trace_PQ = torch.trace(torch.mm(P, Q))
                result[row, i] = trace_PQ
                row += 1
    return result.T

def hadamard_sum(A, B):
    C = A * B
    result = torch.sum(C)
    return result



if __name__ == '__main__':
    """
    References:
        The code framework for training and testing references the DSN[1] code.
        [1]Simon, Christian, et al. "Adaptive subspaces for few-shot learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=3)
    parser.add_argument('--save-path', default=os.path.join(parent_dir, 'best_model', 'NUS_5view'))
    parser.add_argument('--root', default=os.path.join(parent_dir, 'data_process'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lamb', type=float, default=0.03)
    parser.add_argument('--eta', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001)
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
    log_file_path = os.path.join(args.save_path, '{}way-{}shot'.format(args.test_way, args.shot), 'acc.txt')
    sys.stdout = open(log_file_path, 'w')
    save_dir = os.path.join(args.save_path)

    testset = NUS_WIDE_OBJECTdataset(args)
    test_sampler = CategoriesSampler(testset.label, [25, 31], 400,
                                    args.test_way,
                                    args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=0,
                            pin_memory=False)
# NUS_WIDE_OBJECT: 65  226  145  74  129

    model_view1, optimizer_view1, lr_scheduler_1 = create_model(65, [64])
    model_view2, optimizer_view2, lr_scheduler_2 = create_model(226, [64])
    model_view3, optimizer_view3, lr_scheduler_3 = create_model(145, [64])
    model_view4, optimizer_view4, lr_scheduler_4 = create_model(74, [64])
    model_view5, optimizer_view5, lr_scheduler_5 = create_model(129, [1024])

    att1, optimizer_alpha1, lr_scheduler_alpha1 = create_attention(65+226)
    att2, optimizer_alpha2, lr_scheduler_alpha2 = create_attention(65+145)
    att3, optimizer_alpha3, lr_scheduler_alpha3 = create_attention(65+74)
    att4, optimizer_alpha4, lr_scheduler_alpha4 = create_attention(65+129)
    att5, optimizer_alpha5, lr_scheduler_alpha5 = create_attention(226 + 145)
    att6, optimizer_alpha6, lr_scheduler_alpha6 = create_attention(226+74)
    att7, optimizer_alpha7, lr_scheduler_alpha7 = create_attention(226 + 129)
    att8, optimizer_alpha8, lr_scheduler_alpha8 = create_attention(145+74)
    att9, optimizer_alpha9, lr_scheduler_alpha9 = create_attention(145 + 129)
    att10, optimizer_alpha10, lr_scheduler_alpha10 = create_attention(129 + 74)


    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot

    print("Loading best model from validation...")
    # model_view1.load_state_dict(torch.load(os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'epoch=170', 'model_view1_best.pth')))
    # model_view2.load_state_dict(torch.load(os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'epoch=90', 'model_view2_best.pth')))
    # model_view3.load_state_dict(torch.load(os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'epoch=135', 'model_view3_best.pth')))
    # att1.load_state_dict(torch.load(os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'epoch=135', 'att1_best.pth')))
    # att2.load_state_dict(torch.load(os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'epoch=135', 'att2_best.pth')))
    # att3.load_state_dict(torch.load(os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'epoch=135', 'att3_best.pth')))

    model_view1.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'view1model_best.pth')))
    model_view2.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'view2model_best.pth')))
    model_view3.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'view3model_best.pth')))
    model_view4.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'view4model_best.pth')))
    model_view5.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'view5model_best.pth')))
    att1.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att1_best.pth')))
    att2.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att2_best.pth')))
    att3.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att3_best.pth')))
    att4.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att4_best.pth')))
    att5.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att5_best.pth')))
    att6.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att6_best.pth')))
    att7.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att7_best.pth')))
    att8.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att8_best.pth')))
    att9.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att9_best.pth')))
    att10.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.test_way, args.shot), 'att10_best.pth')))


    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)


    trlog = {}
    trlog['args'] = vars(args)
    trlog['test_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['view1_max_acc'] = 0.0
    trlog['view2_max_acc'] = 0.0
    trlog['view3_max_acc'] = 0.0
    trlog['view4_max_acc'] = 0.0
    trlog['view5_max_acc'] = 0.0
    trlog['view1_acc_list'] = []
    trlog['view2_acc_list'] = []
    trlog['view3_acc_list'] = []
    trlog['view4_acc_list'] = []
    trlog['view5_acc_list'] = []
    trlog['fusion_acc_list'] = []

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        torch.cuda.empty_cache()
        model_view1.eval()
        model_view2.eval()
        model_view3.eval()
        model_view4.eval()
        model_view5.eval()

        start_time = time.time()

# test
        tl = Averager()
        tl_view1 = Averager()
        tl_view2 = Averager()
        tl_view3 = Averager()
        tl_view4 = Averager()
        tl_view5 = Averager()

        ta = Averager()
        ta_view1 = Averager()
        ta_view2 = Averager()
        ta_view3 = Averager()
        ta_view4 = Averager()
        ta_view5 = Averager()


        for i, batch in enumerate(test_loader, 1):
            data_view1, data_view2, data_view3, data_view4, data_view5, _= [x.cuda() for x in batch]
            p = args.shot * args.test_way
            data_shot_view1, data_query_view1 = data_view1[:p], data_view1[p:]
            data_shot_view2, data_query_view2 = data_view2[:p], data_view2[p:]
            data_shot_view3, data_query_view3 = data_view3[:p], data_view3[p:]
            data_shot_view4, data_query_view4 = data_view4[:p], data_view4[p:]
            data_shot_view5, data_query_view5 = data_view5[:p], data_view5[p:]


            if args.shot == 1:
                data_shot_view1 = torch.cat((data_shot_view1, flip(data_shot_view1, 3)), dim=0)
                data_shot_view2 = torch.cat((data_shot_view2, flip(data_shot_view2, 3)), dim=0)
                data_shot_view3 = torch.cat((data_shot_view3, flip(data_shot_view3, 3)), dim=0)
                data_shot_view4 = torch.cat((data_shot_view4, flip(data_shot_view4, 3)), dim=0)
                data_shot_view5 = torch.cat((data_shot_view5, flip(data_shot_view5, 3)), dim=0)

            hyperplanes_1, logits_view1, loss_view1, acc_view1 = process_view(data_shot_view1, data_query_view1,
                                                                              model_view1, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_2, logits_view2, loss_view2, acc_view2 = process_view(data_shot_view2, data_query_view2,
                                                                              model_view2, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_3, logits_view3, loss_view3, acc_view3 = process_view(data_shot_view3, data_query_view3,
                                                                              model_view3, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_4, logits_view4, loss_view4, acc_view4 = process_view(data_shot_view4, data_query_view4,
                                                                              model_view4, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_5, logits_view5, loss_view5, acc_view5 = process_view(data_shot_view5, data_query_view5,
                                                                              model_view5, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(
                    logits_view1 + logits_view2 + logits_view3 + logits_view4 + logits_view5, label)



            ta_view1.add(acc_view1)
            ta_view2.add(acc_view2)
            ta_view3.add(acc_view3)
            ta_view4.add(acc_view4)
            ta_view5.add(acc_view5)
            ta.add(acc)


        if ta.item() > trlog['max_acc']:
            trlog['max_acc'] = ta.item()

        if ta_view1.item() > trlog['view1_max_acc']:
            trlog['view1_max_acc'] = ta_view1.item()

        if ta_view2.item() > trlog['view2_max_acc']:
            trlog['view2_max_acc'] = ta_view2.item()

        if ta_view3.item() > trlog['view3_max_acc']:
            trlog['view3_max_acc'] = ta_view3.item()

        if ta_view4.item() > trlog['view4_max_acc']:
            trlog['view4_max_acc'] = ta_view4.item()

        if ta_view5.item() > trlog['view5_max_acc']:
            trlog['view5_max_acc'] = ta_view5.item()

        # if epoch % 10 == 0:
        #     print('view1_epoch {}, test, maxacc={:.4f}'.format(epoch,  trlog['view1_max_acc']))
        #     print('view2_epoch {}, test, maxacc={:.4f}'.format(epoch, trlog['view2_max_acc']))
        #     print('view3_epoch {}, test, maxacc={:.4f}'.format(epoch, trlog['view3_max_acc']))
        #     print('view4_epoch {}, test, maxacc={:.4f}'.format(epoch, trlog['view4_max_acc']))
        #     print('view5_epoch {}, test, maxacc={:.4f}'.format(epoch, trlog['view5_max_acc']))
        #     print('epoch {}, test, maxacc={:.4f}'.format(epoch, trlog['max_acc']))


        trlog['test_acc'].append(ta.item())
        trlog['view1_acc_list'].append(ta_view1.item())
        trlog['view2_acc_list'].append(ta_view2.item())
        trlog['view3_acc_list'].append(ta_view3.item())
        trlog['view4_acc_list'].append(ta_view4.item())
        trlog['view5_acc_list'].append(ta_view5.item())
        trlog['fusion_acc_list'].append(ta.item())

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

        if epoch % 10 == 0:
            # print("\n=== Final Evaluation Summary ===")
            print("Epoch={}".format(epoch))
            print("view1: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['view1_max_acc'],
                statistics.mean(trlog['view1_acc_list']),
                statistics.stdev(trlog['view1_acc_list'])
            ))

            print("view2: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['view2_max_acc'],
                statistics.mean(trlog['view2_acc_list']),
                statistics.stdev(trlog['view2_acc_list'])
            ))

            print("view3: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['view3_max_acc'],
                statistics.mean(trlog['view3_acc_list']),
                statistics.stdev(trlog['view3_acc_list'])
            ))
            print("view4: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['view4_max_acc'],
                statistics.mean(trlog['view4_acc_list']),
                statistics.stdev(trlog['view4_acc_list'])
            ))

            print("view5: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['view5_max_acc'],
                statistics.mean(trlog['view5_acc_list']),
                statistics.stdev(trlog['view5_acc_list'])
            ))

            print("Fusion: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['max_acc'],
                statistics.mean(trlog['fusion_acc_list']),
                statistics.stdev(trlog['fusion_acc_list'])
            ))
