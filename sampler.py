import torch
import numpy as np

class MV_CategoriesSampler():

    def __init__(self, label, n_cls, n_per, n_batch):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)

        unique_labels = np.unique(label)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        remapped_label = np.array([label_mapping[l] for l in label])
        self.m_ind = []
        for i in range(len(unique_labels)):  # 使用新的连续标签范围
            ind = np.argwhere(remapped_label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # self.m_ind = []
        # for i in range(max(label) + 1):
        #     ind = np.argwhere(label == i).reshape(-1)
        #     # print(ind.type)
        #     ind = torch.from_numpy(ind)
        #     self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                if len(l) >= self.n_per:
                    pos = torch.randperm(len(l))[:self.n_per]
                    batch.append(l[pos])
                else:
                    print(f"Warning: Class {c} has fewer than {self.n_per} samples.")
            if batch:
                batch = torch.stack(batch).t().reshape(-1)
                yield batch
            else:
                print(f"Warning: Skipping empty batch.")




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


