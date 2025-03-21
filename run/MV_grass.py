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
from data_process.multiview_dataset import train_dataset, test_dataset
from sampler import MV_CategoriesSampler
from mynet import attention
from subspace_projection import Subspace_Projection, CosineSimilarity
from utils import pprint, set_gpu, Averager, Timer, count_acc, flip, prediction, confusion_matrix
import time


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


def pairwise_trace(P,Q,Z,v):
    assert isinstance(P, torch.Tensor)
    assert isinstance(Q, torch.Tensor)
    assert isinstance(Z, torch.Tensor)
    way = P.shape[0]
    m=3
    result = torch.zeros((m, way)).cuda()
    for i in range(way):
        PP = torch.mm(P[i].t(), P[i])
        QQ = torch.mm(Q[i].t(), Q[i])
        ZZ = torch.mm(Z[i].t(), Z[i])
        trace_PQ = torch.trace(torch.mm(PP, QQ))
        trace_PZ = torch.trace(torch.mm(PP, ZZ))
        trace_QZ = torch.trace(torch.mm(QQ, ZZ))
        result[0, i] = trace_PQ
        result[1, i] = trace_PZ
        result[2, i] = trace_QZ
    return result.T


def hadamard_sum(A, B):
    C = A * B
    result = torch.sum(C)
    return result


def normalize_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    if max_val == min_val:
        return torch.zeros(tensor.shape)

    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor



if __name__ == '__main__':
    """
    References:
        The code framework for training and testing references the DSN[1] code.
        [1]Simon, Christian, et al. "Adaptive subspaces for few-shot learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=3)
    parser.add_argument('--test-way', type=int, default=3)
    # parser.add_argument('--save-path', default='/home/wangxi/dsn_multiview/subspace-3w5s-single') 
    # parser.add_argument('--train-root', default='/home/wangxi/MASA_code/MV_grass_dataset/train_val')
    # parser.add_argument('--test-root', default='/home/wangxi/MASA_code/MV_grass_dataset/test/grass')
    parser.add_argument('--train-root', default=os.path.join(parent_dir, 'MV_grass_dataset', 'train_val'))
    parser.add_argument('--test-root', default=os.path.join(parent_dir, 'MV_grass_dataset', 'test', 'grass'))
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gpu', default='0')
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

    trainset = train_dataset(args)
    train_sampler = MV_CategoriesSampler(trainset.label,
                                      args.train_way,
                                      args.shot + args.query, 100)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

    valset = test_dataset(args)
    val_sampler = MV_CategoriesSampler(valset.label,
                                    args.test_way,
                                    args.shot + args.query, 400)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0,
                            pin_memory=True)


    model_front, optimizer_front, lr_scheduler_front = create_model('resnet18.pth')
    model_side, optimizer_side, lr_scheduler_side = create_model('resnet18.pth')
    model_top, optimizer_top, lr_scheduler_top = create_model('resnet18.pth')

    att1, optimizer_alpha1, lr_scheduler_alpha1 = create_attention(224*224*3 + 224*224*3, [16, 1])
    att2, optimizer_alpha2, lr_scheduler_alpha2 = create_attention(224*224*3 + 224*224*3, [16, 1])
    att3, optimizer_alpha3, lr_scheduler_alpha3 = create_attention(224*224*3 + 224*224*3, [16, 1])

    if args.shot == 1:
        att1, optimizer_alpha1, lr_scheduler_alpha1 = create_attention((224 * 224 * 3 + 224 * 224 * 3)*2, [16, 1])
        att2, optimizer_alpha2, lr_scheduler_alpha2 = create_attention((224 * 224 * 3 + 224 * 224 * 3)*2, [16, 1])
        att3, optimizer_alpha3, lr_scheduler_alpha3 = create_attention((224 * 224 * 3 + 224 * 224 * 3)*2, [16, 1])

    if args.shot == 1:
        shot_num = 2
        args.subspace_dim = 1
    else:
        shot_num = args.shot

    projection_pro = Subspace_Projection(num_dim=args.subspace_dim)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['front_max_acc'] = 0.0
    trlog['side_max_acc'] = 0.0
    trlog['top_max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        torch.cuda.empty_cache()
        lr_scheduler_front.step()
        lr_scheduler_side.step()
        lr_scheduler_top.step()
        if args.adaptive == True:
            lr_scheduler_alpha1.step()
            lr_scheduler_alpha2.step()
            lr_scheduler_alpha3.step()

        model_front.train()
        model_side.train()
        model_top.train()
        if args.adaptive == True:
            att1.train()
            att2.train()
            att3.train()

        tl = Averager()
        tl_front = Averager()
        tl_side = Averager()
        tl_top = Averager()

        ta = Averager()
        ta_front = Averager()
        ta_side = Averager()
        ta_top = Averager()

        start_time = time.time()
        for i, batch in enumerate(train_loader, 1):
            data_front, data_side, data_top, _ = [x.cuda() for x in batch]
            p = args.shot * args.train_way
            qq = p + args.query * args.train_way
            data_shot_front, data_query_front = data_front[:p], data_front[p:qq]
            data_shot_side, data_query_side = data_side[:p], data_side[p:qq]
            data_shot_top, data_query_top = data_top[:p], data_top[p:qq]

            if args.shot == 1:
                data_shot_front = torch.cat((data_shot_front, flip(data_shot_front, 3)), dim=0)
                data_shot_side = torch.cat((data_shot_side, flip(data_shot_side, 3)), dim=0)
                data_shot_top = torch.cat((data_shot_top, flip(data_shot_top, 3)), dim=0)

            if args.adaptive == True:
                view1 = data_shot_front.reshape(args.train_way, shot_num, -1)
                data_1 = torch.cat([view1[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view2 = data_shot_side.reshape(args.train_way, shot_num, -1)
                data_2 = torch.cat([view2[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view3 = data_shot_top.reshape(args.train_way, shot_num, -1)
                data_3 = torch.cat([view3[i].reshape(1, -1) for i in range(args.train_way)], dim=0)


            hyperplanes_1, logits_front, loss_front, acc_front = process_view(data_shot_front, data_query_front,
                                                                              model_front, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_2, logits_side, loss_side, acc_side = process_view(data_shot_side, data_query_side,
                                                                              model_side, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_3, logits_top, loss_top, acc_top = process_view(data_shot_top, data_query_top,
                                                                              model_top, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(logits_front+logits_side+logits_top, label)


            tl_front.add(loss_front.item())
            tl_side.add(loss_side.item())
            tl_top.add(loss_top.item())
            ta.add(acc)


            if args.adaptive == True:
                data12 = torch.cat((data_1, data_2), dim=1)
                data13 = torch.cat((data_1, data_3), dim=1)
                data23 = torch.cat((data_2, data_3), dim=1)
                view12 = att1(data12.to(torch.float))
                view13 = att2(data13.to(torch.float))
                view23 = att3(data23.to(torch.float))
                A = torch.cat((view12,view13,view23), dim=1)
                A = normalize_tensor(A)
                # print(A)
                pairwise = pairwise_trace(hyperplanes_1, hyperplanes_2, hyperplanes_3, v=3)
                loss_pairwise = hadamard_sum(A, pairwise)

            loss = loss_front + loss_side + loss_top - args.eta * loss_pairwise

            optimizer_front.zero_grad()
            optimizer_side.zero_grad()
            optimizer_top.zero_grad()
            optimizer_alpha1.zero_grad()
            optimizer_alpha2.zero_grad()
            optimizer_alpha3.zero_grad()
            loss.backward()
            optimizer_front.step()
            optimizer_side.step()
            optimizer_top.step()
            optimizer_alpha1.step()
            optimizer_alpha2.step()
            optimizer_alpha3.step()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch}/{args.max_epoch} - Duration: {epoch_duration:.2f} seconds")

        print('front_epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch, loss_front.item(), acc_front))
        print('side_epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch, loss_side.item(), acc_side))
        print('top_epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch, loss_top.item(), acc_top))
        print('epoch {},   acc={:.4f}'
              .format(epoch, acc))

        tl = tl.item()
        ta = ta.item()

        if epoch < 100 and epoch % 3 != 0:
             continue
        with torch.no_grad():
            model_front.eval()
            model_side.eval()
            model_top.eval()
            att1.eval()
            att2.eval()
            att3.eval()



# TEST
            vl = Averager()
            vl_front = Averager()
            vl_side = Averager()
            vl_top = Averager()

            va = Averager()
            va_front = Averager()
            va_side = Averager()
            va_top = Averager()
            conf_matrix = torch.zeros(args.test_way, args.test_way)

            for i, batch in enumerate(val_loader, 1):
                data_front, data_side, data_top, _ = [x.cuda() for x in batch]
                p = args.shot * args.test_way
                data_shot_front, data_query_front = data_front[:p], data_front[p:]
                data_shot_side, data_query_side = data_side[:p], data_side[p:]
                data_shot_top, data_query_top = data_top[:p], data_top[p:]

                if args.shot == 1:
                    data_shot_front = torch.cat((data_shot_front, flip(data_shot_front, 3)), dim=0)
                    data_shot_side = torch.cat((data_shot_side, flip(data_shot_side, 3)), dim=0)
                    data_shot_top = torch.cat((data_shot_top, flip(data_shot_top, 3)), dim=0)


                hyperplanes_1, logits_front, loss_front, acc_front = process_view(data_shot_front, data_query_front,
                                                                                  model_front, shot_num, args.train_way,
                                                                                  args.query, args.lamb)
                hyperplanes_2, logits_side, loss_side, acc_side = process_view(data_shot_side, data_query_side,
                                                                               model_side, shot_num, args.train_way,
                                                                               args.query, args.lamb)
                hyperplanes_3, logits_top, loss_top, acc_top = process_view(data_shot_top, data_query_top,
                                                                            model_top, shot_num, args.train_way,
                                                                            args.query, args.lamb)
                label = torch.arange(args.train_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                acc = count_acc(logits_front + logits_side + logits_top, label)

                vl_front.add(loss_front)
                vl_side.add(loss_side)
                vl_top.add(loss_top)
                va_front.add(acc_front)
                va_side.add(acc_side)
                va_top.add(acc_top)

                va.add(acc)
                conf_matrix = confusion_matrix(logits_front + logits_side + logits_top, label, conf_matrix)


            if va_front.item() > trlog['front_max_acc']:
                trlog['front_max_acc'] = va_front.item()

            if va_side.item() > trlog['side_max_acc']:
                trlog['side_max_acc'] = va_side.item()

            if va_top.item() > trlog['top_max_acc']:
                trlog['top_max_acc'] = va_top.item()

            if va.item() > trlog['max_acc']:
                trlog['max_acc'] = va.item()

            if epoch > 50 and epoch % 10 != 0:
                print('front_epoch {}, val, front_maxacc={:.4f}'.format(epoch, trlog['front_max_acc']))
                print('side_epoch {}, val, side_maxacc={:.4f}'.format(epoch, trlog['side_max_acc']))
                print('top_epoch {}, val, top_maxacc={:.4f}'.format(epoch, trlog['top_max_acc']))
                print('fusion_epoch {}, val,  maxacc={:.4f}'.format(epoch, trlog['max_acc']))
                # print(conf_matrix)

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl.item())
            trlog['val_acc'].append(va.item())

            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
