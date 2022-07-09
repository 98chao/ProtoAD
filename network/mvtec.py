import os
import numpy as np
from PIL import Image
from sklearn import datasets

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):

    def __init__(self, root='/home/huangchao/Huang/EmbeddingAD/Data', class_name='bottle', is_train=True, resize=256, cropsize=224):
        super().__init__()
        self.root = root
        self.class_name = class_name
        self.is_train = is_train
        self.crop_size = cropsize

        assert self.class_name in CLASS_NAMES, "classname must be in {}".format(CLASS_NAMES)

        self.img_path, self.labels, self.mask_path, self.types, self.ids = self.load_dataset()
        
        if is_train:
            self.transform_img = T.Compose([
                T.Resize(resize, Image.ANTIALIAS),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform_img = T.Compose([
                T.Resize(resize, Image.ANTIALIAS),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])

        self.transform_mask = T.Compose([
            T.Resize(resize, Image.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor()
        ])

    def load_dataset(self):
        flag = "train" if self.is_train else "test"
        img_path, labels, mask_path, types_path, ids = [], [], [], [], []

        img_dir = os.path.join(self.root, self.class_name, flag)
        mask_dir = os.path.join(self.root, self.class_name, "ground_truth")
        types = sorted(os.listdir(img_dir))

        for _type in types:
            img_dir_type = os.path.join(img_dir, _type)
            if not os.path.isdir(img_dir_type):
                continue
            # load img path
            img_path_list = sorted([os.path.join(img_dir_type, f) for f in os.listdir(img_dir_type) if f.endswith(".png")])
            img_path.extend(img_path_list)

            # load labels and mask
            if _type == "good":
                labels.extend([0] * len(img_path_list))
                mask_path.extend([None] * len(img_path_list))
                # ids.extend(i for i in range(len(img_path_list)))
                ids.extend(str(path.split('/')[-2])+"_"+str(path.split('/')[-1]) for path in img_path_list)
            else:
                labels.extend([1] * len(img_path_list))
                mask_type_dir = os.path.join(mask_dir, _type)
                img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                mask_path_list = [os.path.join(mask_type_dir, img_name + "_mask.png") for img_name in img_name_list]
                mask_path.extend(mask_path_list)
                # ids.extend([0] * len(img_path_list))
                ids.extend(str(path.split('/')[-2])+"_"+str(path.split('/')[-1]) for path in img_path_list)
            
            # load types
            types_path.extend([_type] * len(img_path_list))
            
        assert (len(img_path) == len(labels) and len(labels) == len(ids)), "the number of img and labels should be same"

        return list(img_path), list(labels), list(mask_path), list(types_path), list(ids)
    
    def __getitem__(self, idx):
        image, label, mask, img_id = self.img_path[idx], self.labels[idx], self.mask_path[idx], self.ids[idx]
        if self.is_train:
            img_id = image.split('/')[-1].split('.')[0]    
        
        image = Image.open(image).convert("RGB")
        image = self.transform_img(image)
        
        if label == 0:
            mask = torch.zeros([1, self.crop_size, self.crop_size])
        else:
            mask = Image.open(mask).convert("L")
            mask = self.transform_mask(mask)

        return image, label, mask, img_id

    def __len__(self):
        return len(self.img_path)
        

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * np.array(std)) + np.array(mean)) * 255.).astype(np.uint8)
    return x
