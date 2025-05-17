import argparse
import numpy as np
import random
import os
import sys
# Get the absolute path of a script
script_path = os.path.abspath(sys.argv[0])
# Get the previous directory
parent_dir = os.path.dirname(os.path.dirname(script_path))
# Add the upper level directory to sys.path
sys.path.append(parent_dir)

import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_process.multiview_dataset import test_dataset
from sampler import MV_CategoriesSampler
from mynet import attention
from subspace_projection import Subspace_Projection, CosineSimilarity
from utils import pprint, set_gpu, Averager, Timer, count_acc, flip, prediction, confusion_matrix, normalize_tensor, pairwise_trace, hadamard_sum
import time
import statistics


def create_model(model_path, requires_grad=['fc'], step_size1=30, step_size2=100):
    model = torch.load(model_path).cuda()
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = (name in requires_grad)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    step_size = step_size1 if args.shot > 1 else step_size2
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    return model, optimizer, lr_scheduler


def create_attention(input_size, output_size):
    att = attention(input_size * args.shot, output_size).cuda().float()
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




if __name__ == '__main__':
    """
    References:
        The code framework for training and testing references the DSN[1] code.
        [1]Simon, Christian, et al. "Adaptive subspaces for few-shot learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=3)
    parser.add_argument('--test-way', type=int, default=3)
    parser.add_argument('--save-path', default=os.path.join(parent_dir, 'best_model', 'MV_grass'))
    parser.add_argument('--train-root', default=os.path.join(parent_dir, 'MV_grass_dataset', 'train_val'))
    parser.add_argument('--test-root', default=os.path.join(parent_dir, 'MV_grass_dataset', 'test'))
    # parser.add_argument('--test-root',
    #                     default=os.path.join("/media/imin/DATA/wangxi/MASA/MULT_dataset/pre_process/test/paper"))
    parser.add_argument('--seed', type=int, default=1111, help='random seed')   # 2222,3333
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--lamb', type=float, default=0.03)
    parser.add_argument('--eta', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--envs', type=str, default='grass')
    parser.add_argument('--adaptive', action='store_true', default=True)

    args = parser.parse_args()
    args.subspace_dim = args.shot-1

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

    log_file_path = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot), 'acc-paper.txt')
    sys.stdout = open(log_file_path, 'w')
    save_dir = os.path.join(args.save_path)

    testset = test_dataset(args)
    test_sampler = MV_CategoriesSampler(testset.label, args.test_way, args.shot + args.query, 400)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=0, pin_memory=True)



    model_front, _, _ = create_model('resnet18.pth')
    model_side, _, _ = create_model('resnet18.pth')
    model_top, _, _ = create_model('resnet18.pth')

    att1, _, _ = create_attention(224*224*3 + 224*224*3, [16, 1])
    att2, _, _ = create_attention(224*224*3 + 224*224*3, [16, 1])
    att3, _, _ = create_attention(224*224*3 + 224*224*3, [16, 1])

    if args.shot == 1:
        att1, _, _  = create_attention((224 * 224 * 3 + 224 * 224 * 3)*2, [16, 1])
        att2, _, _  = create_attention((224 * 224 * 3 + 224 * 224 * 3)*2, [16, 1])
        att3, _, _  = create_attention((224 * 224 * 3 + 224 * 224 * 3)*2, [16, 1])

    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot

    testset = test_dataset(args)
    test_sampler = MV_CategoriesSampler(testset.label, args.test_way, args.shot + args.query, 400)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=0, pin_memory=True)

    print("Loading best model from validation...")
    model_front.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.train_way, args.shot), 'model_front_best.pth')))
    model_side.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.train_way, args.shot),  'model_side_best.pth')))
    model_top.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.train_way, args.shot),  'model_top_best.pth')))
    att1.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.train_way, args.shot),  'att1_best.pth')))
    att2.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.train_way, args.shot), 'att2_best.pth')))
    att3.load_state_dict(torch.load(
        os.path.join(save_dir, '{}way-{}shot'.format(args.train_way, args.shot), 'att3_best.pth')))

    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['max_acc'] = 0.0
    trlog['front_max_acc'] = 0.0
    trlog['side_max_acc'] = 0.0
    trlog['top_max_acc'] = 0.0
    trlog['front_acc_list'] = []
    trlog['side_acc_list'] = []
    trlog['top_acc_list'] = []
    trlog['fusion_acc_list'] = []

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        torch.cuda.empty_cache()

        model_front.eval()
        model_side.eval()
        model_top.eval()
        att1.eval()
        att2.eval()
        att3.eval()


        tl = Averager()
        tl_front = Averager()
        tl_side = Averager()
        tl_top = Averager()

        ta = Averager()
        ta_front = Averager()
        ta_side = Averager()
        ta_top = Averager()
        conf_matrix = torch.zeros(args.test_way, args.test_way)

        for i, batch in enumerate(test_loader, 1):
            data_front, data_side, data_top, _ = [x.cuda() for x in batch]
            p = args.shot * args.test_way
            data_shot_front, data_query_front = data_front[:p], data_front[p:]
            data_shot_side, data_query_side = data_side[:p], data_side[p:]
            data_shot_top, data_query_top = data_top[:p], data_top[p:]

            if args.shot == 1:
                data_shot_front = torch.cat((data_shot_front, flip(data_shot_front, 3)), dim=0)
                data_shot_side = torch.cat((data_shot_side, flip(data_shot_side, 3)), dim=0)
                data_shot_top = torch.cat((data_shot_top, flip(data_shot_top, 3)), dim=0)

            # print(data_shot_front.shape)
            hyperplanes_1, logits_front, loss_front, acc_front = process_view(data_shot_front, data_query_front,
                                                                              model_front, shot_num, args.test_way,
                                                                              args.query, args.lamb)
            hyperplanes_2, logits_side, loss_side, acc_side = process_view(data_shot_side, data_query_side,
                                                                           model_side, shot_num, args.test_way,
                                                                           args.query, args.lamb)
            hyperplanes_3, logits_top, loss_top, acc_top = process_view(data_shot_top, data_query_top,
                                                                        model_top, shot_num, args.test_way,
                                                                        args.query, args.lamb)
            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(logits_front + logits_side + logits_top, label)

            tl_front.add(loss_front)
            tl_side.add(loss_side)
            tl_top.add(loss_top)
            ta_front.add(acc_front)
            ta_side.add(acc_side)
            ta_top.add(acc_top)

            ta.add(acc)
            conf_matrix = confusion_matrix(logits_front + logits_side + logits_top, label, conf_matrix)


        if ta_front.item() > trlog['front_max_acc']:
            trlog['front_max_acc'] = ta_front.item()

        if ta_side.item() > trlog['side_max_acc']:
            trlog['side_max_acc'] = ta_side.item()

        if ta_top.item() > trlog['top_max_acc']:
            trlog['top_max_acc'] = ta_top.item()

        if ta.item() > trlog['max_acc']:
            trlog['max_acc'] = ta.item()


        # if epoch % 5 != 0:
        print('front_epoch {}, test_acc={:.4f}, front_maxacc={:.4f}'.format(epoch, ta_front.item(), trlog['front_max_acc']))
        print('side_epoch {}, test_acc={:.4f}, side_maxacc={:.4f}'.format(epoch, ta_side.item(), trlog['side_max_acc']))
        print('top_epoch {}, test_acc={:.4f}, top_maxacc={:.4f}'.format(epoch, ta_top.item(), trlog['top_max_acc']))
        print('fusion_epoch {}, test_acc={:.4f},  maxacc={:.4f}'.format(epoch, ta.item(), trlog['max_acc']))
            # print(conf_matrix)


        trlog['front_acc_list'].append(ta_front.item())
        trlog['side_acc_list'].append(ta_side.item())
        trlog['top_acc_list'].append(ta_top.item())
        trlog['fusion_acc_list'].append(ta.item())

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

        if epoch % 10 == 0:
            # print("\n=== Final Evaluation Summary ===")
            print("Epoch={}".format(epoch))
            print("Front View: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['front_max_acc'],
                statistics.mean(trlog['front_acc_list']),
                statistics.stdev(trlog['front_acc_list'])
            ))

            print("Side View: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['side_max_acc'],
                statistics.mean(trlog['side_acc_list']),
                statistics.stdev(trlog['side_acc_list'])
            ))

            print("Top View: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['top_max_acc'],
                statistics.mean(trlog['top_acc_list']),
                statistics.stdev(trlog['top_acc_list'])
            ))

            print("Fusion: max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
                trlog['max_acc'],
                statistics.mean(trlog['fusion_acc_list']),
                statistics.stdev(trlog['fusion_acc_list'])
            ))