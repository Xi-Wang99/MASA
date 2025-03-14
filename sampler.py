import torch
import numpy as np

class MV_CategoriesSampler():

    def __init__(self, label, n_cls, n_per, n_batch):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)

        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            # print(ind.type)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  #这样处理后，每个batch中的n_class都是随机的
            for c in classes:  #对抽出的类循环
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]  #n_per是每个类抽的样本数
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch




class CategoriesSampler():

    def __init__(self, label, label_index, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label.cpu())
        self.m_ind = []
        for i in range(label_index[0], label_index[1]):
            ind = np.argwhere(label == i + 1).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                c_true = c
                l = self.m_ind[c_true]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch