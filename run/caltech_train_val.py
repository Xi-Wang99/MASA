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


def pairwise_trace(P,Q,Z,v):
    assert isinstance(P, torch.Tensor)
    assert isinstance(Q, torch.Tensor)
    assert isinstance(Z, torch.Tensor)

    way = P.shape[0]

    if v==3:
        m=3
    if v==4:
        m=6
    if v==5:
        m=10
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


def create_model(input_size, hidden_sizes):
    model = net(input_size, hidden_sizes).cuda().float()
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
    # print(data_shot.shape)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=3)
    parser.add_argument('--save-path', default=os.path.join(parent_dir, 'best_model', 'caltech'))
    parser.add_argument('--root', default=os.path.join(parent_dir, 'data_process'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lamb', type=float, default=0.03)
    parser.add_argument('--eta', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=1111, help='random seed') #2222, 3333
    parser.add_argument('--low', action='store_true', default=False)
    parser.add_argument('--high', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--adaptive', action='store_true', default=False)

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


    trainset = Caltechdataset(args)
    train_sampler = CategoriesSampler(trainset.label, [0, 12], 100,
                                      args.train_way,
                                      args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=False)

    valset = Caltechdataset(args)
    val_sampler = CategoriesSampler(valset.label, [12, 16], 100,
                                    args.train_way,
                                    args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0,
                            pin_memory=False)

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


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['view1_max_acc'] = 0.0
    trlog['view2_max_acc'] = 0.0
    trlog['view3_max_acc'] = 0.0


    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        torch.cuda.empty_cache()
        lr_scheduler_1.step()
        lr_scheduler_2.step()
        lr_scheduler_3.step()

        if args.adaptive == True:
            lr_scheduler_alpha1.step()
            lr_scheduler_alpha2.step()
            lr_scheduler_alpha3.step()

        model_view1.train()
        model_view2.train()
        model_view3.train()

        if args.adaptive == True:
            att1.train()
            att2.train()
            att3.train()

        tl = Averager()
        tl_view1 = Averager()
        tl_view2 = Averager()
        tl_view3 = Averager()

        ta = Averager()
        ta_view1 = Averager()
        ta_view2 = Averager()
        ta_view3 = Averager()


        for i, batch in enumerate(train_loader, 1):
            if args.low == True:
                data_view1, data_view2, data_view3, _, _, _, _ = [x.cuda() for x in batch]
            elif args.high == True:
                _, _, _, data_view1, data_view2, data_view3, _ = [x.cuda() for x in batch]
            p = args.shot * args.train_way
            qq = p + args.query * args.train_way
            data_shot_view1, data_query_view1 = data_view1[:p], data_view1[p:qq]
            data_shot_view2, data_query_view2 = data_view2[:p], data_view2[p:qq]
            data_shot_view3, data_query_view3 = data_view3[:p], data_view3[p:qq]



            if args.adaptive == True:
                view1 = data_shot_view1.reshape(args.train_way, shot_num, -1)
                data_1 = torch.cat([view1[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view2 = data_shot_view2.reshape(args.train_way, shot_num, -1)
                data_2 = torch.cat([view2[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view3 = data_shot_view3.reshape(args.train_way, shot_num, -1)
                data_3 = torch.cat([view3[i].reshape(1, -1) for i in range(args.train_way)], dim=0)

            hyperplanes_1, logits_view1, loss_view1, acc_view1 = process_view(data_shot_view1, data_query_view1,
                                                                              model_view1, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            # print(data_query_view1.shape)
            # print(hyperplanes_1.shape)
            hyperplanes_2, logits_view2, loss_view2, acc_view2 = process_view(data_shot_view2, data_query_view2,
                                                                              model_view2, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_3, logits_view3, loss_view3, acc_view3 = process_view(data_shot_view3, data_query_view3,
                                                                              model_view3, shot_num, args.train_way,
                                                                              args.query, args.lamb)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(logits_view1 + logits_view2 + logits_view3, label)



            tl_view1.add(loss_view1.item())
            tl_view2.add(loss_view2.item())
            tl_view3.add(loss_view3.item())

            ta.add(acc)

            if args.adaptive == True:
                data12 = torch.cat((data_1, data_2), dim=1)
                data13 = torch.cat((data_1, data_3), dim=1)
                data23 = torch.cat((data_2, data_3), dim=1)
                view12 = att1(data12.to(torch.float))
                view13 = att2(data13.to(torch.float))
                view23 = att3(data23.to(torch.float))
                A = torch.cat((view12, view13, view23), dim=1)
                pairwise = pairwise_trace(hyperplanes_1, hyperplanes_2, hyperplanes_3, v=3)
                loss_pairwise = hadamard_sum(A, pairwise)
            else:
                loss_pairwise = 0

            loss = loss_view1 + loss_view2 + loss_view3 - args.eta * loss_pairwise

            optimizer_view1.zero_grad()
            optimizer_view2.zero_grad()
            optimizer_view3.zero_grad()
            optimizer_alpha1.zero_grad()
            optimizer_alpha2.zero_grad()
            optimizer_alpha3.zero_grad()
            loss.backward()
            optimizer_view1.step()
            optimizer_view2.step()
            optimizer_view3.step()
            optimizer_alpha1.step()
            optimizer_alpha2.step()
            optimizer_alpha3.step()


        print('view1_epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch, loss_view1.item(), acc_view1))
        print('view2_epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch, loss_view2.item(), acc_view2))
        print('view3_epoch {},   loss={:.4f} acc={:.4f}'
              .format(epoch, loss_view3.item(), acc_view3))
        print('epoch {},   co_acc={:.4f}'
              .format(epoch,  acc))

        tl = tl.item()
        ta = ta.item()

        if epoch % 5 != 0:
            continue

        with torch.no_grad():
            model_view1.eval()
            model_view2.eval()
            model_view3.eval()
            att1.eval()
            att2.eval()
            att3.eval()


# validation
        vl = Averager()
        vl_view1 = Averager()
        vl_view2 = Averager()
        vl_view3 = Averager()

        va = Averager()
        va_view1 = Averager()
        va_view2 = Averager()
        va_view3 = Averager()


        for i, batch in enumerate(val_loader, 1):
            if args.low == True:
                data_view1, data_view2, data_view3, _, _, _, _ = [x.cuda() for x in batch]
            elif args.high == True:
                _, _, _, data_view1, data_view2, data_view3, _ = [x.cuda() for x in batch]
            p = args.shot * args.train_way
            data_shot_view1, data_query_view1 = data_view1[:p], data_view1[p:]
            data_shot_view2, data_query_view2 = data_view2[:p], data_view2[p:]
            data_shot_view3, data_query_view3 = data_view3[:p], data_view3[p:]
            #print(data_query_view1.shape)


            hyperplanes_1, logits_view1, loss_view1, acc_view1 = process_view(data_shot_view1, data_query_view1,
                                                                              model_view1, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_2, logits_view2, loss_view2, acc_view2 = process_view(data_shot_view2, data_query_view2,
                                                                              model_view2, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_3, logits_view3, loss_view3, acc_view3 = process_view(data_shot_view3, data_query_view3,
                                                                              model_view3, shot_num, args.train_way,
                                                                              args.query, args.lamb)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(logits_view1 + logits_view2 + logits_view3, label)

            vl_view1.add(loss_view1)
            vl_view2.add(loss_view2)
            vl_view3.add(loss_view3)

            va_view1.add(acc_view1)
            va_view2.add(acc_view2)
            va_view3.add(acc_view3)
            va.add(acc)

        print('view1_epoch {}, val, loss={:.4f} acc={:.4f} maxacc={:.4f}'.format(epoch,vl_view1.item(), va_view1.item(), trlog['view1_max_acc']))
        print('view2_epoch {}, val, loss={:.4f} acc={:.4f} maxacc={:.4f}'.format(epoch, vl_view2.item(),  va_view2.item(),  trlog['view2_max_acc']))
        print('view3_epoch {}, val, loss={:.4f} acc={:.4f} maxacc={:.4f}'.format(epoch,  vl_view3.item(),   va_view3 .item(),  trlog['view3_max_acc']))
        print('epoch {}, val, loss={:.4f} acc={:.4f} maxacc={:.4f}'.format(epoch, vl.item(), va.item(), trlog['max_acc']))


        if args.low:
            agg_name = 'low'
        elif args.high:
            agg_name = 'high'
        if va.item() > trlog['max_acc']:
            trlog['max_acc'] = va.item()
            save_dir = os.path.join(args.save_path, agg_name, '{}way-{}shot'.format(args.train_way, args.shot))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(att1.state_dict(), os.path.join(save_dir, 'att1_best.pth'))
            torch.save(att2.state_dict(), os.path.join(save_dir, 'att2_best.pth'))
            torch.save(att3.state_dict(), os.path.join(save_dir, 'att3_best.pth'))
        if va_view1.item() > trlog['view1_max_acc']:
            trlog['view1_max_acc'] = va_view1.item()
            save_dir = os.path.join(args.save_path, agg_name, '{}way-{}shot wo'.format(args.train_way, args.shot))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model_view1.state_dict(), os.path.join(save_dir, 'view1model_best.pth'))
        if va_view2.item() > trlog['view2_max_acc']:
            trlog['view2_max_acc'] = va_view2.item()
            save_dir = os.path.join(args.save_path, agg_name, '{}way-{}shot wo'.format(args.train_way, args.shot))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model_view2.state_dict(), os.path.join(save_dir, 'view2model_best.pth'))
        if va_view3.item() > trlog['view3_max_acc']:
            trlog['view3_max_acc'] = va_view3.item()
            save_dir = os.path.join(args.save_path, agg_name, '{}way-{}shot wo'.format(args.train_way, args.shot))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model_view3.state_dict(), os.path.join(save_dir, 'view3model_best.pth'))


        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl.item())
        trlog['val_acc'].append(va.item())

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))


