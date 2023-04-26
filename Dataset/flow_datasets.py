import numpy as np
from torch.utils.data import Dataset
import glob
from PIL import Image


class UnsupervisedOpticalDataset(Dataset):
    def __init__(self, root, transform=None, co_transform=None,
                 target_transform=None, ap_transform=None):
        super(UnsupervisedOpticalDataset, self).__init__()
        self.root = root
        self.input_transform = transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        self.img_list = glob.glob(root + '/*reference.tif')
        self.img_list.sort()
        self.n_frames = 2

    def __getitem__(self, index):
        img_reference = self.img_list[index]
        img_moving = img_reference.replace('reference', 'moving')
        img_reference = Image.open(img_reference).convert('RGB')
        img_moving = Image.open(img_moving).convert('RGB')
        images = [np.array(img_reference), np.array(img_moving)]
        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            images, _ = self.co_transform(images, {})
        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]
        data = {'img{}'.format(i + 1): p for i, p in enumerate(images)}

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
            for i in range(self.n_frames):
                data['img{}_ph'.format(i + 1)] = imgs_ph[i]

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = self.target_transform[key](target[key])
        return data

    def __len__(self):
        return len(self.img_list)


class UnsupervisedOpticalDatasetValid(UnsupervisedOpticalDataset):
    def __getitem__(self, index):
        img_reference = self.img_list[index]
        img_moving = img_reference.replace('reference', 'moving')
        img_gt = img_reference.replace('reference', 'gt')
        img_reference = Image.open(img_reference).convert('RGB')
        img_moving = Image.open(img_moving).convert('RGB')
        img_gt = Image.open(img_gt).convert('RGB')
        data = {'img1': self.input_transform(np.array(img_reference)),
                'img2': self.input_transform(np.array(img_moving)),
                'img_gt': self.input_transform(np.array(img_gt))}
        return data

    def __len__(self):
        return len(self.img_list)
