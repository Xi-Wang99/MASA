import torch.nn as nn
import torch
import scipy.linalg
import numpy as np

class CosineSimilarity(nn.Module):

    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


class Subspace_Projection(nn.Module):
    """
    Subspace mapping module.
    References:
        The create_subspace function and the projection_metric function codes is modified based on the DSN[1] functions.
        [1]Simon, Christian, et al. "Adaptive subspaces for few-shot learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
    """
    def __init__(self, num_dim=3):
        super().__init__()
        self.num_dim = num_dim

    def create_subspace(self, supportset_features, class_size, sample_size):  #定义一个如何计算每类超平面和平均值的函数
        all_hyper_planes = [] #超平面
        means = []   #每类的平均值
        for ii in range(class_size):   #循环处理每一个类
            num_sample = sample_size
            all_support_within_class_t = supportset_features[ii]#all_support_within_class_t=[f_theta(x_c)]=[f_theta(x_c,1),f_theta(x_c,2)……f_theta(x_c,k)]

            meann = torch.mean(all_support_within_class_t, dim=0)
            means.append(meann)
            all_support_within_class_t = all_support_within_class_t - meann.unsqueeze(0).repeat(num_sample, 1)

            all_support_within_class = torch.transpose(all_support_within_class_t, 0, 1)
            # all_support_within_class = torch.where(torch.isnan(all_support_within_class), torch.full_like(all_support_within_class, 0), all_support_within_class)

            try:
                uu, s, v = torch.linalg.svd(all_support_within_class.double(), full_matrices=False)
            except:  # torch.linalg.svd may have convergence issues for GPU and CPU.
                uu, s, v = torch.linalg.svd(
                    (all_support_within_class.double() + 1e-3 * torch.rand_like(all_support_within_class)),
                    full_matrices=False)

            uu = uu.float()

            all_hyper_planes.append(uu[:, :self.num_dim])

        all_hyper_planes = torch.stack(all_hyper_planes, dim=0)  #Splice the all_hyper_planes tensor in dimension 0  [way,48,subspace]
        # print(all_hyper_planes.shape)
        means = torch.stack(means)  #Splicing the means [way,48]
        if len(all_hyper_planes.size()) < 3:
            all_hyper_planes = all_hyper_planes.unsqueeze(-1)
        return all_hyper_planes, means



    def projection_metric(self, target_features, hyperplanes, mu):
        eps = 1e-12
        batch_size = target_features.shape[0]
        class_size = hyperplanes.shape[0]
        similarities = []
        discriminative_loss = 0.0

        for j in range(class_size):
            h_plane_j = hyperplanes[j].unsqueeze(0).repeat(batch_size, 1, 1)   #[6,1024.subspace]
            target_features_expanded = (target_features - mu[j].expand_as(target_features)).unsqueeze(-1)   #f_theta(q)-mu_c
            projected_query_j = torch.bmm(h_plane_j, torch.bmm(torch.transpose(h_plane_j, 1, 2), target_features_expanded))
            projected_query_j = torch.squeeze(projected_query_j) + mu[j].unsqueeze(0).repeat(batch_size, 1)
            projected_query_dist_inter = target_features - projected_query_j

            # Training per epoch is slower but less epochs in total
            query_loss = -torch.sqrt(
                torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) + eps)  # norm ||.||  d_c(q)
            # print(query_loss.shape)


            # Training per epoch is faster but more epochs in total
            # query_loss = -torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) # Squared norm ||.||^2

            similarities.append(query_loss)


            #discriminative_loss
            for k in range(class_size):
                if j != k:
                    temp_loss = torch.mm(torch.transpose(hyperplanes[j], 0, 1), hyperplanes[k])
                    # discriminative subspaces (Conv4 only, ResNet12 is computationally expensive)
                    discriminative_loss = discriminative_loss + torch.sum(temp_loss * temp_loss)

        similarities = torch.stack(similarities, dim=1)
        # print(similarities.shape)

        return similarities, discriminative_loss





    # def logits(self, query, support):   #定义一个query图片到该类超平面距离的度量
    #     eps = 1e-12
    #     batch_size = query.shape[0]      #这个不确定，记得回来复盘！! #矩阵0维的个数(行数)，有几个query就是多大的batch size
    #     class_size = support.shape[0]   #超平面0维的个数（行数），有几个超平面就有几个类
    #     rank = support.shape[1]
    #     # print(rank)
    #
    #
    #     if rank == 1:
    #         similarities = torch.zeros(15, 3)
    #         def EuclideanDistances(a, b):
    #             sq_a = a ** 2
    #             sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    #             sq_b = b ** 2
    #             sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    #             bt = b.t()
    #             return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))
    #
    #         for i in range(class_size):
    #             for j in range(batch_size):
    #                 similarities[j,i] = EuclideanDistances(query[j], support[i])
    #         # print(similarities)
    #     else:
    #         similarities = []  # 相似度矩阵
    #         for j in range(class_size):
    #             h_plane_j = support[j].unsqueeze(0).repeat(batch_size, 1, 1) .transpose(1, 2)  # 在第1维重复batch size次 #[6,1024.1]
    #             # print(h_plane_j.shape)  # [15,512,4]
    #             projected_query_j = torch.bmm(h_plane_j, torch.bmm(h_plane_j.transpose(1, 2), query.transpose(1, 2)))
    #             # print(projected_query_j.shape)  # [15,512,1]
    #             # print(torch.bmm(h_plane_j, query_expanded))
    #             projected_query_j = torch.squeeze(projected_query_j)
    #             # print(projected_query_j.shape)  # [15,512]
    #             projected_query_dist_inter = query.squeeze() - projected_query_j
    #             # print(projected_query_dist_inter.shape)
    #             # print(projected_query_j)
    #
    #             # Training per epoch is slower but less epochs in total
    #             query_loss = -torch.sqrt(
    #                 torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=1) + eps)  # norm ||.||  #也就是d_c(q)
    #             # print(query_loss.shape)
    #             # Training per epoch is faster but more epochs in total
    #             # query_loss = -torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) # Squared norm ||.||^2
    #             similarities.append(query_loss)
    #
    #         similarities = torch.stack(similarities, dim=1).cuda()
    #         # print(similarities.shape)
    #
    #         # pdist = nn.PairwiseDistance(p=2)
    #         # for i in range(class_size):
    #         #     for j in range(batch_size):
    #         #         similarities[j,i] = pdist(support[i],query[j])
    #         #         j = j+1
    #         #     i = i+1
    #
    #     return similarities











