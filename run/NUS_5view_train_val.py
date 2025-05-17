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
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=3)
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


    trainset = NUS_WIDE_OBJECTdataset(args)

    train_sampler = CategoriesSampler(trainset.label, [0, 15], 100,
                                      args.train_way,
                                      args.shot + args.query)  # 100batch
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=False)

    valset = NUS_WIDE_OBJECTdataset(args)
    val_sampler = CategoriesSampler(valset.label, [15, 25], 400,
                                    args.train_way,
                                    args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
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
    trlog['view4_max_acc'] = 0.0
    trlog['view5_max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        torch.cuda.empty_cache()

        lr_scheduler_1.step()
        lr_scheduler_2.step()
        lr_scheduler_3.step()
        lr_scheduler_4.step()
        lr_scheduler_5.step()
        if args.adaptive == True:
            lr_scheduler_alpha1.step()
            lr_scheduler_alpha2.step()
            lr_scheduler_alpha3.step()
            lr_scheduler_alpha4.step()
            lr_scheduler_alpha5.step()
            lr_scheduler_alpha6.step()
            lr_scheduler_alpha7.step()
            lr_scheduler_alpha8.step()
            lr_scheduler_alpha9.step()
            lr_scheduler_alpha10.step()


        model_view1.train()
        model_view2.train()
        model_view3.train()
        model_view4.train()
        model_view5.train()
        if args.adaptive == True:
            att1.train()
            att2.train()
            att3.train()
            att4.train()
            att5.train()
            att6.train()
            att7.train()
            att8.train()
            att9.train()
            att10.train()


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

        start_time = time.time()
        for i, batch in enumerate(train_loader, 1):
            data_view1, data_view2, data_view3, data_view4, data_view5,  _ = [x.cuda() for x in batch]
            p = args.shot * args.train_way
            qq = p + args.query * args.train_way
            data_shot_view1, data_query_view1 = data_view1[:p], data_view1[p:qq]
            data_shot_view2, data_query_view2 = data_view2[:p], data_view2[p:qq]
            data_shot_view3, data_query_view3 = data_view3[:p], data_view3[p:qq]
            data_shot_view4, data_query_view4 = data_view4[:p], data_view4[p:qq]
            data_shot_view5, data_query_view5 = data_view5[:p], data_view5[p:qq]

            data_view1 = data_shot_view1.reshape(1, -1).squeeze()
            data_view2 = data_shot_view2.reshape(1, -1).squeeze()
            data_view3 = data_shot_view3.reshape(1, -1).squeeze()
            data_view4 = data_shot_view4.reshape(1, -1).squeeze()
            data_view5 = data_shot_view5.reshape(1, -1).squeeze()

            if args.shot == 1:
                data_shot_view1 = torch.cat((data_shot_view1, flip(data_shot_view1, 3)), dim=0)
                data_shot_view2 = torch.cat((data_shot_view2, flip(data_shot_view2, 3)), dim=0)
                data_shot_view3 = torch.cat((data_shot_view3, flip(data_shot_view3, 3)), dim=0)
                data_shot_view4 = torch.cat((data_shot_view4, flip(data_shot_view4, 3)), dim=0)
                data_shot_view5 = torch.cat((data_shot_view5, flip(data_shot_view5, 3)), dim=0)

            if args.adaptive == True:
                view1 = data_shot_view1.reshape(args.train_way, shot_num, -1)
                data_1 = torch.cat([view1[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view2 = data_shot_view2.reshape(args.train_way, shot_num, -1)
                data_2 = torch.cat([view2[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view3 = data_shot_view3.reshape(args.train_way, shot_num, -1)
                data_3 = torch.cat([view3[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view4 = data_shot_view4.reshape(args.train_way, shot_num, -1)
                data_4 = torch.cat([view4[i].reshape(1, -1) for i in range(args.train_way)], dim=0)
                view5 = data_shot_view5.reshape(args.train_way, shot_num, -1)
                data_5 = torch.cat([view5[i].reshape(1, -1) for i in range(args.train_way)], dim=0)


            hyperplanes_1, logits_view1, loss_view1, acc_view1 = process_view(data_shot_view1, data_query_view1,
                                                                              model_view1, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_2, logits_view2, loss_view2, acc_view2 = process_view(data_shot_view2, data_query_view2,
                                                                              model_view2, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_3, logits_view3, loss_view3, acc_view3 = process_view(data_shot_view3, data_query_view3,
                                                                              model_view3, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_4, logits_view4, loss_view4, acc_view4 = process_view(data_shot_view4, data_query_view4,
                                                                              model_view4, shot_num, args.train_way,
                                                                              args.query, args.lamb)
            hyperplanes_5, logits_view5, loss_view5, acc_view5 = process_view(data_shot_view5, data_query_view5,
                                                                              model_view5, shot_num, args.train_way,
                                                                              args.query, args.lamb)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            acc = count_acc(logits_view1 + logits_view2 + logits_view3 + logits_view4 + logits_view5, label)


            tl_view1.add(loss_view1.item())
            tl_view2.add(loss_view2.item())
            tl_view3.add(loss_view3.item())
            tl_view4.add(loss_view4.item())
            tl_view5.add(loss_view5.item())

            ta.add(acc)

            if args.adaptive == True:
                data12 = torch.cat((data_1, data_2), dim=1)
                data13 = torch.cat((data_1, data_3), dim=1)
                data14 = torch.cat((data_1, data_4), dim=1)
                data15 = torch.cat((data_1, data_5), dim=1)
                data23 = torch.cat((data_2, data_3), dim=1)
                data24 = torch.cat((data_2, data_4), dim=1)
                data25 = torch.cat((data_2, data_5), dim=1)
                data34 = torch.cat((data_3, data_4), dim=1)
                data35 = torch.cat((data_3, data_5), dim=1)
                data45 = torch.cat((data_4, data_5), dim=1)

                view12 = att1(data12.to(torch.float))
                view13 = att2(data13.to(torch.float))
                view14 = att3(data14.to(torch.float))
                view15 = att4(data15.to(torch.float))
                view23 = att5(data23.to(torch.float))
                view24 = att6(data24.to(torch.float))
                view25 = att7(data25.to(torch.float))
                view34 = att8(data34.to(torch.float))
                view35 = att9(data35.to(torch.float))
                view45 = att10(data45.to(torch.float))
                A = torch.cat((view12, view13, view14, view15, view23, view24, view25, view34, view35, view45), dim=1)
                pairwise = pairwise_5view(hyperplanes_1, hyperplanes_2, hyperplanes_3, hyperplanes_4, hyperplanes_5)
                loss_pairwise = hadamard_sum(A, pairwise)

            loss = loss_view1 + loss_view2 + loss_view3 + loss_view4 + loss_view5 - args.eta * loss_pairwise


            optimizer_view1.zero_grad()
            optimizer_view2.zero_grad()
            optimizer_view3.zero_grad()
            optimizer_view4.zero_grad()
            optimizer_view5.zero_grad()
            optimizer_alpha1.zero_grad()
            optimizer_alpha2.zero_grad()
            optimizer_alpha3.zero_grad()
            optimizer_alpha4.zero_grad()
            optimizer_alpha5.zero_grad()
            optimizer_alpha6.zero_grad()
            optimizer_alpha7.zero_grad()
            optimizer_alpha8.zero_grad()
            optimizer_alpha9.zero_grad()
            optimizer_alpha10.zero_grad()


            loss.backward()
            optimizer_view1.step()
            optimizer_view2.step()
            optimizer_view3.step()
            optimizer_view4.step()
            optimizer_view5.step()
            optimizer_alpha1.step()
            optimizer_alpha2.step()
            optimizer_alpha3.step()
            optimizer_alpha4.step()
            optimizer_alpha5.step()
            optimizer_alpha6.step()
            optimizer_alpha7.step()
            optimizer_alpha8.step()
            optimizer_alpha9.step()
            optimizer_alpha10.step()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch}/{args.max_epoch} - Duration: {epoch_duration:.2f} seconds")
        print('view1_epoch {},   loss={:.4f} acc={:.4f}'
          .format(epoch, loss_view1.item(), acc_view1))
        print('view2_epoch {},   loss={:.4f} acc={:.4f}'
          .format(epoch, loss_view2.item(), acc_view2))
        print('view3_epoch {},   loss={:.4f} acc={:.4f}'
          .format(epoch, loss_view3.item(), acc_view3))
        print('view4_epoch {},   loss={:.4f} acc={:.4f}'
          .format(epoch, loss_view4.item(), acc_view4))
        print('view5_epoch {},   loss={:.4f} acc={:.4f}'
          .format(epoch, loss_view5.item(), acc_view5))
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
            model_view4.eval()
            model_view5.eval()
            att1.eval()
            att2.eval()
            att3.eval()
            att4.eval()
            att5.eval()
            att6.eval()
            att7.eval()
            att8.eval()
            att9.eval()
            att10.eval()
            projection_pro.eval()


# VAL
            vl = Averager()
            vl_view1 = Averager()
            vl_view2 = Averager()
            vl_view3 = Averager()
            vl_view4 = Averager()
            vl_view5 = Averager()

            va = Averager()
            va_view1 = Averager()
            va_view2 = Averager()
            va_view3 = Averager()
            va_view4 = Averager()
            va_view5 = Averager()


            for i, batch in enumerate(val_loader, 1):
                data_view1, data_view2, data_view3, data_view4, data_view5, _= [x.cuda() for x in batch]
                p = args.shot * args.train_way
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
                                                                                  model_view1, shot_num, args.train_way,
                                                                                  args.query, args.lamb)
                hyperplanes_2, logits_view2, loss_view2, acc_view2 = process_view(data_shot_view2, data_query_view2,
                                                                                  model_view2, shot_num, args.train_way,
                                                                                  args.query, args.lamb)
                hyperplanes_3, logits_view3, loss_view3, acc_view3 = process_view(data_shot_view3, data_query_view3,
                                                                                  model_view3, shot_num, args.train_way,
                                                                                  args.query, args.lamb)
                hyperplanes_4, logits_view4, loss_view4, acc_view4 = process_view(data_shot_view4, data_query_view4,
                                                                                  model_view4, shot_num, args.train_way,
                                                                                  args.query, args.lamb)
                hyperplanes_5, logits_view5, loss_view5, acc_view5 = process_view(data_shot_view5, data_query_view5,
                                                                                  model_view5, shot_num, args.train_way,
                                                                                  args.query, args.lamb)

                acc = count_acc(
                        logits_view1 + logits_view2 + logits_view3 + logits_view4 + logits_view5, label)


                vl_view1.add(loss_view1)
                vl_view2.add(loss_view2)
                vl_view3.add(loss_view3)
                vl_view4.add(loss_view4)
                vl_view5.add(loss_view5)

                va_view1.add(acc_view1)
                va_view2.add(acc_view2)
                va_view3.add(acc_view3)
                va_view4.add(acc_view4)
                va_view5.add(acc_view5)
                va.add(acc)



            if va.item() > trlog['max_acc']:
                trlog['max_acc'] = va.item()
                save_dir = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(att1.state_dict(), os.path.join(save_dir, 'att1_best.pth'))
                torch.save(att2.state_dict(), os.path.join(save_dir, 'att2_best.pth'))
                torch.save(att3.state_dict(), os.path.join(save_dir, 'att3_best.pth'))
                torch.save(att4.state_dict(), os.path.join(save_dir, 'att4_best.pth'))
                torch.save(att5.state_dict(), os.path.join(save_dir, 'att5_best.pth'))
                torch.save(att6.state_dict(), os.path.join(save_dir, 'att6_best.pth'))
                torch.save(att7.state_dict(), os.path.join(save_dir, 'att7_best.pth'))
                torch.save(att8.state_dict(), os.path.join(save_dir, 'att8_best.pth'))
                torch.save(att9.state_dict(), os.path.join(save_dir, 'att9_best.pth'))
                torch.save(att10.state_dict(), os.path.join(save_dir, 'att10_best.pth'))
            if va_view1.item() > trlog['view1_max_acc']:
                trlog['view1_max_acc'] = va_view1.item()
                save_dir = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model_view1.state_dict(), os.path.join(save_dir, 'view1model_best.pth'))
            if va_view2.item() > trlog['view2_max_acc']:
                trlog['view2_max_acc'] = va_view2.item()
                save_dir = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model_view2.state_dict(), os.path.join(save_dir, 'view2model_best.pth'))
            if va_view3.item() > trlog['view3_max_acc']:
                trlog['view3_max_acc'] = va_view3.item()
                save_dir = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model_view3.state_dict(), os.path.join(save_dir, 'view3model_best.pth'))
            if va_view4.item() > trlog['view4_max_acc']:
                trlog['view4_max_acc'] = va_view4.item()
                save_dir = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model_view4.state_dict(), os.path.join(save_dir, 'view4model_best.pth'))
            if va_view5.item() > trlog['view5_max_acc']:
                trlog['view5_max_acc'] = va_view5.item()
                save_dir = os.path.join(args.save_path, '{}way-{}shot'.format(args.train_way, args.shot))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model_view5.state_dict(), os.path.join(save_dir, 'view5model_best.pth'))


            if epoch % 10 == 0:
                print('view1_epoch {}, val, maxacc={:.4f}'.format(epoch,  trlog['view1_max_acc']))
                print('view2_epoch {}, val, maxacc={:.4f}'.format(epoch, trlog['view2_max_acc']))
                print('view3_epoch {}, val, maxacc={:.4f}'.format(epoch, trlog['view3_max_acc']))
                print('view4_epoch {}, val, maxacc={:.4f}'.format(epoch, trlog['view4_max_acc']))
                print('view5_epoch {}, val, maxacc={:.4f}'.format(epoch, trlog['view5_max_acc']))
                print('epoch {}, val, maxacc={:.4f}'.format(epoch, trlog['max_acc']))

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl.item())
            trlog['val_acc'].append(va.item())

            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))