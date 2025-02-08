import logging
import math

import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data

from augmentation import RandAugment
import random
import copy

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
aff_mean = (0.5863, 0.4595, 0.4030)
aff_std = (0.2715, 0.2424, 0.2366)

def get_data(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])

    # AffectNet_image_path = "/home/mnt/Dataset/AffectNet/AffectNet/tra_rename/Manually_Annotated_images"
    AffectNet_image_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/AffectNet/tra_rename/Manually_Annotated_images"

    # AffectNet_label_path = "/home/mnt/Dataset/AffectNet/AffectNet/affect_train_sameAs_RAF.txt"
    AffectNet_label_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/AffectNet/affect_train_sameAs_RAF.txt"
    
    # RAF_image_path = "/home/mnt/Dataset/RAF/train_RAF"
    RAF_image_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/RAF/train_RAF"
    # RAF_label_path = "/home/mnt/Dataset/RAF/labels_train_rename.txt"
    RAF_label_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/RAF/labels_train_rename.txt"

    # test_image_path = "/home/mnt/Dataset/RAF/test_RAF"
    test_image_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/RAF/test_RAF"
    # test_label_path = "/home/mnt/Dataset/RAF/labels_test_rename.txt"
    test_label_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/RAF/labels_test_rename.txt"

    # RAF_label_path_train_t = "/home/mnt/Dataset/RAF/labels_train_rename_new_train_t.txt"
    RAF_label_path = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/RAF/labels_train_rename.txt"
   
    RAF_label_path_val = "/mnt/beegfs/home/cv/dai.guan/lyq/Master/RAF/labels_train_rename_val_balance2.txt"

    label_dataset = RAFData(image_path = RAF_image_path, label_path=RAF_label_path, transform=transform_labeled)

    val_dataset = RAFData(image_path=RAF_image_path, label_path=RAF_label_path_val, transform=transform_val)
   
    unlabel_dataset = AffectData(image_path=AffectNet_image_path, label_path = AffectNet_label_path,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
   
    test_dataset = RAFData(image_path=test_image_path, label_path=test_label_path, transform=transform_val)
    
    all_label_dataset = RAFData(image_path = RAF_image_path, label_path=RAF_label_path, transform=transform_labeled)
    
    label_dataset_t = label_dataset
    label_dataset_s = label_dataset

    val_dataset_t = val_dataset
    val_dataset_s = val_dataset

    # return label_dataset_t,unlabel_dataset, test_dataset


    
    return label_dataset_t,label_dataset_s, unlabel_dataset, test_dataset,val_dataset_t, val_dataset_s, all_label_dataset


def get_raf_data(args):
    transform_labeled = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=aff_mean, std=aff_std)
    ])

    path_label='../data/raf_train.csv'
    path_unlabel='../data/raf_aff_train.csv'
    path_test='../data/raf_test.csv'
    DF_label=pd.read_csv(path_label)
    DF_unlabel=pd.read_csv(path_unlabel)
    DF_test=pd.read_csv(path_test)
    label_dataset_dir='/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    unlabel_dataset_dir='/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    test_dataset_dir='/home/data/lzy/AffectNet/Manually_Annotated/Manually_Annotated_Images'
    label_dataset = RafData(data=DF_label, directory=label_dataset_dir, transform=transform_labeled)
    unlabel_dataset = RafData(data=DF_unlabel, directory=unlabel_dataset_dir,
                             transform=TransformMPL(args, mean=aff_mean, std=aff_std))
    test_dataset = RafData(data=DF_test, directory=test_dataset_dir, transform=transform_val)
    return label_dataset, unlabel_dataset, test_dataset


class AffectData(data.Dataset):
    def __init__(self, image_path, label_path, transform):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.image_list = []
        self.label_list = []
        label_file = open(label_path, 'r')
        for line in label_file.readlines():
            directory_id = line.strip().split('_')[2]
            image_item_name = line.strip().split(' ')[0]
            image_item_path = os.path.join(self.image_path, directory_id, image_item_name)
            self.image_list.append(image_item_path)
            self.label_list.append(-1)
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_item_path = self.image_list[idx]
        image = Image.open(image_item_path).convert('RGB')
        label = int(self.label_list[idx]) # 注意label需要为整数, 否则会报错
        img = self.transform(image)
        return img, label

class RAFData(data.Dataset):
    def __init__(self, image_path, label_path, transform):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.image_list = []
        self.label_list = []
        label_file = open(label_path, 'r')
        for line in label_file.readlines():
            image_item_name = line.strip().split(' ')[0]
            label = line.strip().split(' ')[1]
            image_item_path = os.path.join(self.image_path, image_item_name)
            self.image_list.append(image_item_path)
            self.label_list.append(label)
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_item_path = self.image_list[idx]
        image = Image.open(image_item_path).convert('RGB')
        label = int(self.label_list[idx])
        img = self.transform(image)
        return img,label

class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 10, 10  # default

        self.ori = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(args.resize),
            transforms.Resize(256),
            transforms.RandomCrop(args.resize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=args.resize,
            #               padding=int(args.resize*0.125),
            #                 padding_mode='reflect')
#             transforms.CenterCrop(args.resize)
        ])
        self.aug = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(args.resize),
            transforms.Resize(256),
            transforms.RandomCrop(args.resize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=args.resize,
            #                       padding=int(args.resize*0.125),
            #                      padding_mode='reflect'),
#             transforms.CenterCrop(args.resize),
            RandAugment(n=n, m=m)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


DATASET_GETTERS = {'get_data': get_data,
             'get_raf_data': get_raf_data}
