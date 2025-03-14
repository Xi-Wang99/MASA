from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import argparse

class train_dataset(Dataset):
    def __init__(self, args):

        data_root = args.train_root
        data_root1 = data_root + '/front'
        data_root2 = data_root + '/side'
        data_root3 = data_root + '/top'

        # self.cut = args.cut
        test_transform = transforms.Compose([
            #transforms.CenterCrop((self.cut, self.cut)),
            transforms.Resize((224, 224), interpolation=2),  #功能：重置图像分辨率为224*224。interpolation-插值方法选择，默认为PIL.Image.BILINEAR
            transforms.ToTensor(),  #图片变成[3,224,224]的张量形式
            #transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        label_str = []
        imgs_front = []
        imgs_side = []
        imgs_top = []
        label_str2 = []
        label_str3 = []


        for root, dirs, files in os.walk(data_root1):  # root 表示当前正在访问的文件夹路径；dirs 表示该文件夹下的子目录名list；files 表示该文件夹下的文件list
            files.sort() #将文件排序
            for file in files:
                label_str.append(file.split('_')[0])   #只记录文件名'_'以前的内容作为标签，例如bb,bf
                img_front = test_transform(Image.open(data_root1 + '/' + file))
                imgs_front.append(np.array(img_front))  #把front图像都放在imgs_front = []中
        for root, dirs, files in os.walk(data_root2):
            files.sort()
            for file in files:
                label_str2.append(file.split('_')[0])
                img_side = test_transform(Image.open(data_root2 + '/' + file))
                imgs_side.append(np.array(img_side))
        for root, dirs, files in os.walk(data_root3):
            files.sort()
            for file in files:
                label_str3.append(file.split('_')[0])
                img_top = test_transform(Image.open(data_root3 + '/' + file))
                imgs_top.append(np.array(img_top))

        #print('front label = side label', label_str == label_str2)
        #print('front label = top label', label_str == label_str3)
        imgs_front = torch.tensor(imgs_front)
        imgs_side = torch.tensor(imgs_side)
        imgs_top = torch.tensor(imgs_top)
        #把图集合转化成张量

        label = []
        temp = 0
        tmp = 0
        for index in range(len(label_str)):
            if index == 0:
                clas = 0
                label.append(clas)

            else:
                for j in range(index):
                    if label_str[index] == label_str[index - j - 1]:
                        clas = label[index - j - 1]
                        label.append(clas)
                        tmp = 0
                        break
                    else:
                        tmp += 1
                if tmp == index and tmp != 0:
                    temp += 1
                    tmp = 0
                    clas = temp
                    label.append(clas)
            #print('index', index, 'label', label[index], 'tmp', tmp)
        label = [int(x) for x in label] #把lable里面cls都化为整数形式
        label = torch.tensor(label)   #把lable化为张量

        self.imgs_front = imgs_front
        self.imgs_side = imgs_side
        self.imgs_top = imgs_top
        self.label = label

    def __getitem__(self, i):
        imgs_front, imgs_side, imgs_top, label = self.imgs_front[i], self.imgs_side[i], self.imgs_top[i], self.label[i]
        return imgs_front, imgs_side, imgs_top, label


class test_dataset(Dataset):
    def __init__(self,args):
        super(test_dataset, self).__init__()
        roots = args.test_root
        # roots = '/home/liujing/dataset/MULT_dataset/center_crop_800/test/grass'
        # roots = '/home/liujing/datasets/MULT_test'
        self.envs = args.envs
        # self.cut = args.cut

        test_transform = transforms.Compose([
            # transforms.CenterCrop((self.cut, self.cut)),
            transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [.5, .5 ,.5],std = [.5, .5 ,.5])
        ])

        label_str = []
        label_str2 = []
        label_str3 = []
        imgs_front = []
        imgs_side = []
        imgs_top = []

        for root, dirs, files in os.walk(roots):
            files.sort()
            if root == os.path.join(roots, 'front'):
            # if root == os.path.join(roots, 'crop_front'):
                for file in files:
                    class_name = file.split('_')[0]
                    if self.envs == 'grass':
                        if class_name == 'bb' or class_name == 'bf' or class_name == 'bt' or class_name == 'bm':
                            label_str.append(class_name)
                            img_front = test_transform(Image.open(root + '/' + file))
                            imgs_front.append(np.array(img_front))
                    else:
                        if class_name == 'b' or class_name == 'f' or class_name == 't' or class_name == 'm':
                            label_str.append(class_name)
                            img_front = test_transform(Image.open(root + '/' + file))
                            imgs_front.append(np.array(img_front))

            if root == os.path.join(roots, 'side'):
            # if root == os.path.join(roots, 'crop_side'):
                for file in files:
                    class_name = file.split('_')[0]
                    if self.envs == 'grass':
                        if class_name == 'bb' or class_name == 'bf' or class_name == 'bt' or class_name == 'bm':
                            label_str2.append(class_name)
                            img_side = test_transform(Image.open(root + '/' + file))
                            imgs_side.append(np.array(img_side))
                    else:
                        if class_name == 'b' or class_name == 'f' or class_name == 't' or class_name == 'm':
                            label_str2.append(file.split('_')[0])
                            img_side = test_transform(Image.open(root + '/' + file))
                            imgs_side.append(np.array(img_side))

            if root == os.path.join(roots, 'top'):
            # if root == os.path.join(roots, 'crop_top'):
                for file in files:
                    class_name = file.split('_')[0]
                    if self.envs == 'grass':
                        if class_name == 'bb' or class_name == 'bf' or class_name == 'bt' or class_name == 'bm':
                            label_str3.append(file.split('_')[0])
                            img_top = test_transform(Image.open(root + '/' + file))
                            imgs_top.append(np.array(img_top))
                    else:
                        if class_name == 'b' or class_name == 'f' or class_name == 't' or class_name == 'm':
                            label_str3.append(file.split('_')[0])
                            img_top = test_transform(Image.open(root + '/' + file))
                            imgs_top.append(np.array(img_top))

        imgs_front = torch.tensor(imgs_front)
        imgs_side = torch.tensor(imgs_side)
        imgs_top = torch.tensor(imgs_top)
        # print('front label = side label', label_str == label_str2)
        # print('front label = top label', label_str == label_str3)
        # print('front label',label_str)
        label = []
        temp = 0
        tmp = 0
        for index in range(len(label_str)):
            if index == 0:
                clas = 0
                label.append(clas)
                '''elif label_str[index] == label_str[index - 1]:
                clas = 0
                label.append(clas)
            else:
                clas += 1
                label.append(clas)'''
            else:
                for j in range(index):
                    if label_str[index] == label_str[index - j - 1]:
                        clas = label[index - j - 1]
                        label.append(clas)
                        tmp = 0
                        break
                    else:
                        tmp += 1
                if tmp == index and tmp != 0:
                    temp += 1
                    tmp = 0
                    clas = temp
                    label.append(clas)
            # print('index', index, 'label', label[index], 'tmp', tmp)
        label = [int(x) for x in label]
        label = torch.tensor(label)

        self.imgs_front = imgs_front
        self.imgs_side = imgs_side
        self.imgs_top = imgs_top
        self.label = label

    def __getitem__(self, i):
        imgs_front, imgs_side, imgs_top, label = self.imgs_front[i], self.imgs_side[i], self.imgs_top[i], self.label[i]
        return imgs_front, imgs_side, imgs_top, label



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--save-epoch', type=int, default=100)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='/subspace-5w5s-discriminative')  # 记得补上
    parser.add_argument('--train-root', default='/MULT_dataset/pre_process/train_val')
    parser.add_argument('--test-root', default='/MULT_dataset/pre_process/test')  # 用MULT-DATA
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lamb', type=float, default=0.03)  # lamb是loss里面的lamda
    parser.add_argument('--envs', type=str, default='grass')

    args = parser.parse_args()
    #print('imgs_front',train_dataset().imgs_front)
    from torchvision.utils import save_image
    dataside = test_dataset(args).imgs_side
    to_img = transforms.ToPILImage()
    #imgside = to_img(dataside[0] * 0.5 + 0.5)
    for i in range(45):
        save_image(dataside[i],'./side/side[i].png')
    print('side',to_img(dataside[0] * 0.5 + 0.5))
