import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io as sio
import torchvision.transforms as tr
import random
import pandas as pd

def crop(image, label):
    # output_size = (48, 48, 48)
    # output_size = (96, 96, 96)
    output_size = image.shape
    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
            output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape
    # if np.random.uniform() > 0.33:
    #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
    #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
    # else:
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])
    d1 = np.random.randint(0, d - output_size[2])

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    # print (np.max(label))

    return image, label


def flip(image, label):
    # Random H flip
    if random.random() > 0.5:
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
    return image, label


def centercrop(image, label):
    # output_size = (48, 48, 48)
    output_size = image.shape
    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
            output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape

    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
    d1 = int(round((d - output_size[2]) / 2.))

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

    if (np.max(label)) > 1:
        print(np.max(label))

    return image, label

import scipy.ndimage.interpolation as scy
def rotate(image, label):
    angle_list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    random.shuffle(angle_list)
    image = scy.rotate(image, angle=angle_list[0], axes=(1, 2), reshape=False)
    label = scy.rotate(label, angle=angle_list[0], axes=(1, 2), reshape=False)
    label = np.round(label)

    return image, label

class PulmonuryNodule(Dataset):
    # def __init__(self, num_classes=2, nodule_list_csv=None, data_path=None, label_path=None, pat_lung_bbox_csv=None,is_train=True, **kwargs):

    def __init__(self, num_classes=2, nodule_list_csv=None, data_path=None, label_path=None, is_train=True, is_HEM=False, nodule_location_list=None, **kwargs):
        super(PulmonuryNodule, self).__init__(**kwargs)
        self.nodule_list = np.array(pd.read_csv(nodule_list_csv))
        self.num_classes = num_classes
        self.data_path = data_path
        self.label_path = label_path
        self.is_train = is_train
        # if is_HEM:
        #     nodule_list_hem = []
        #     for nodule in self.nodule_list:
        #         if nodule[7] == 1:
        #             for i in range(4):
        #                 nodule_list_hem.append(np.array([nodule[0], nodule[1], nodule[2], nodule[3], nodule[4], nodule[5], nodule[6], nodule[7]]))
        #         elif nodule[7] == 2:
        #             for i in range(8):
        #                 nodule_list_hem.append(np.array([nodule[0], nodule[1], nodule[2], nodule[3], nodule[4], nodule[5], nodule[6], nodule[7]]))
        #         else:
        #             nodule_list_hem.append(np.array([nodule[0], nodule[1], nodule[2], nodule[3], nodule[4], nodule[5], nodule[6], nodule[7]]))
        #     random.shuffle(nodule_list_hem)
        #     self.nodule_list = np.array(nodule_list_hem)

    def __len__(self):
        # if not self.is_train:
        #     return self.nodule_list_new.shape[0]
        # else:
        #     return self.nodule_list_new.shape[0]
        return self.nodule_list.shape[0]

    def __getitem__(self, idx):
        # if not self.is_train:
            # random.shuffle(list(self.nodule_list))
        # self.nodule_list = np.array(self.nodule_list)
        nodule = self.nodule_list[idx, :]
        name = nodule[1]
        coordx = nodule[2]
        coordy = nodule[3]
        coordz = nodule[4]
        hospital = nodule[6]
        date = nodule[7]

        # if hospital == 'LIDC-IDRI':
        #     l = len(name)
        #     for file in os.listdir('/home2/LUNG_DATA/LIDC-IDRI/2018_07_21'):
        #         if file[:l] == name:
        #             name = file
        #             break
        # print(nodule)
        # print(self.data_path+hospital+'/'+date+'/'+name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy')
        image = np.load(self.data_path+hospital+'/'+date+'/'+name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy')
        label = np.load(self.label_path+hospital+'/'+date+'/'+name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy')


        k = random.randint(0,4)
        if k ==0:
            image, label = flip(image, label)
        elif k == 1:
            image, label = crop(image, label)
        elif k == 2:
            image, label = image, label #centercrop(image, label)
        else:
            image, label = rotate(image, label)

        x = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # x = (x-np.min(x))/(np.max(x)-np.min(x))
        y = torch.from_numpy(label.astype(np.float32))
        y = torch.unsqueeze(y, 0)
        # print(y.size())

        # onehot_y = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        # for i in range(self.num_classes):
        #     onehot_y[i, :, :, :] = (label == i).astype(np.float32)
        # y = torch.from_numpy(onehot_y)

        # print(np.max(onehot_y[1,:,:,:]))
        nodule= {}
        nodule['name'] = name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy'
        nodule['image'] = torch.from_numpy(x)
        nodule['label'] = y

        return nodule

class PulmonuryNodule_loss(Dataset):
    # def __init__(self, num_classes=2, nodule_list_csv=None, data_path=None, label_path=None, pat_lung_bbox_csv=None,is_train=True, **kwargs):

    def __init__(self, num_classes=2, nodule_list_csv=None, data_path=None, label_path=None, is_train=True, is_HEM=True, nodule_location_list=None, **kwargs):
        super(PulmonuryNodule_loss, self).__init__(**kwargs)
        self.nodule_list = np.array(pd.read_csv(nodule_list_csv))
        self.num_classes = num_classes
        self.data_path = data_path
        self.label_path = label_path
        self.is_train = is_train
        self.nodule_loc_list = np.array(pd.read_csv(nodule_location_list))
        if is_HEM:
            nodule_list_hem = []
            for nodule in self.nodule_list:
                if nodule[7] == 1:
                    for i in range(4):
                        nodule_list_hem.append(np.array([nodule[0], nodule[1], nodule[2], nodule[3], nodule[4], nodule[5], nodule[6], nodule[7]]))
                elif nodule[7] == 2:
                    for i in range(8):
                        nodule_list_hem.append(np.array([nodule[0], nodule[1], nodule[2], nodule[3], nodule[4], nodule[5], nodule[6], nodule[7]]))
                else:
                    nodule_list_hem.append(np.array([nodule[0], nodule[1], nodule[2], nodule[3], nodule[4], nodule[5], nodule[6], nodule[7]]))
            random.shuffle(nodule_list_hem)
            self.nodule_list = np.array(nodule_list_hem)

    def __len__(self):
        # if not self.is_train:
        #     return self.nodule_list_new.shape[0]
        # else:
        #     return self.nodule_list_new.shape[0]
        return self.nodule_list.shape[0]

    def __getitem__(self, idx):
        # if not self.is_train:
            # random.shuffle(list(self.nodule_list))
        # self.nodule_list = np.array(self.nodule_list)
        nodule = self.nodule_list[idx, :]
        name = nodule[1]
        hospital = nodule[6]
        date = nodule[7]
        coordx = nodule[2]
        coordy = nodule[3]
        coordz = nodule[4]
        nodule_bbox = self.nodule_loc_list[self.nodule_loc_list[:,0]==name]
        nodule_bbox = nodule_bbox[nodule_bbox[:,1] == coordx]
        nodule_bbox = nodule_bbox[nodule_bbox[:,2] == coordy]
        nodule_bbox = nodule_bbox[nodule_bbox[:,3] == coordz]
        z_min, y_min, x_min, z_max, y_max, x_max = nodule_bbox[0,4], nodule_bbox[0,5], nodule_bbox[0,6], nodule_bbox[0,7], nodule_bbox[0,8], nodule_bbox[0,9]
        x_center, y_center, z_center = nodule_bbox[0,10], nodule_bbox[0, 11], nodule_bbox[0, 12]
        # if hospital == 'LIDC-IDRI':
        #     l = len(name)
        #     for file in os.listdir('/home2/LUNG_DATA/LIDC-IDRI/2018_07_21'):
        #         if file[:l] == name:
        #             name = file
        #             break

        image = np.load(self.data_path+hospital+'/'+date+'/'+name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy')
        label = np.load(self.label_path+hospital+'/'+date+'/'+name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy')

        loss_map = np.zeros_like(image)
        x_smallpatch_min = max(0, x_center-24)
        x_smallpatch_max = min(96, x_center+24)
        if x_smallpatch_min == 0:
            x_smallpatch_max = 48
        if x_smallpatch_max == 96:
            x_smallpatch_min = 48

        y_smallpatch_min = max(0, y_center-24)
        y_smallpatch_max = min(96, y_center+24)
        if y_smallpatch_min == 0:
            y_smallpatch_max = 48
        if y_smallpatch_max == 96:
            y_smallpatch_min = 48

        z_smallpatch_min = max(0, z_center - 24)
        z_smallpatch_max = min(96, z_center + 24)
        if z_smallpatch_min == 0:
            z_smallpatch_max = 48
        if z_smallpatch_max == 96:
            z_smallpatch_min = 48

        # print(x_smallpatch_min, x_smallpatch_max, y_smallpatch_min, y_smallpatch_max, z_smallpatch_min, z_smallpatch_max)
        loss_map[z_smallpatch_min:z_smallpatch_max, y_smallpatch_min:y_smallpatch_max, x_smallpatch_min:x_smallpatch_max] = 1
        loss_map = torch.from_numpy(loss_map.astype(np.float32))
        loss_map = torch.unsqueeze(loss_map, 0)

        k = random.randint(0,4)
        if k ==0:
            image, label = flip(image, label)
        elif k == 1:
            image, label = crop(image, label)
        elif k == 2:
            image, label = centercrop(image, label)
        else:
            image, label = rotate(image, label)

        x = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # x = (x-np.min(x))/(np.max(x)-np.min(x))
        y = torch.from_numpy(label.astype(np.float32))
        y = torch.unsqueeze(y, 0)
        # print(y.size())

        # onehot_y = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        # for i in range(self.num_classes):
        #     onehot_y[i, :, :, :] = (label == i).astype(np.float32)
        # y = torch.from_numpy(onehot_y)

        # print(np.max(onehot_y[1,:,:,:]))
        nodule= {}
        nodule['name'] = name+'_'+str(coordx)+'_'+str(coordy)+'_'+str(coordz)+'.npy'
        nodule['image'] = torch.from_numpy(x)
        nodule['label'] = y
        nodule['loss_map'] = loss_map

        return nodule