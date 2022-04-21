import os
import random
from PIL import Image

import cv2
import torch
import numpy as np
import imageio

from src.data.resize_right import resize


def search(root, target='JPEG'):
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            item_list.extend(search(path, target))
        elif path.split('.')[-1] == target:
            item_list.append(path)
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
    return item_list


def get_patch_img(img, patch_size=96, scale=2):
    ih, iw = img.shape[:2]
    tp = scale * patch_size
    if (iw - tp) > -1 and (ih-tp) > 1:
        ix = random.randrange(0, iw-tp+1)
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, ix:ix+tp, :3]
    elif (iw - tp) > -1 and (ih - tp) <= -1:
        ix = random.randrange(0, iw-tp+1)
        hr = img[:, ix:ix+tp, :3]
        pil_img = Image.fromarray(hr).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    elif (iw - tp) <= -1 and (ih - tp) > -1:
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, :, :3]
        pil_img = Image.fromarray(hr).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    else:
        pil_img = Image.fromarray(img).resize((tp, tp), Image.BILINEAR)
        hr = np.array(pil_img)
    return hr


def compress(img, quality_factor):
    img_l = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img_l, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img_l = cv2.imdecode(encimg, 3)
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

    return img_l


class ImageNet():
    def __init__(self, args, train=True, name='ImageNet'):
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        self.dataroot = args.dir_data
        self.img_list = search(os.path.join(self.dataroot, 'imagenet', 'train'), 'JPEG')
        self.img_list.extend(search(os.path.join(self.dataroot, 'imagenet', 'val'), 'JPEG'))
        self.img_list = sorted(self.img_list)
        self.train = train
        self.args = args
        self.len = len(self.img_list)

        print(f'Dataset includes {self.len} number of images')

        self.task = args.task

        self.quality_factor = args.quality_factor

    def __len__(self):
        return len(self.img_list)

    def _get_index(self, idx):
        return idx % len(self.img_list)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.img_list[idx]
        lr = imageio.imread(f_lr)
        if len(lr.shape) == 2:
            lr = np.dstack([lr, lr, lr])
        return lr, f_lr

    def _np2Tensor(self, img, rgb_range):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx % self.len)
        pair = self.get_patch(hr, scale=self.scale)

        if self.task == 'sr':
            pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
            low = resize(pair_t, scale_factors=1/self.scale)
            
        elif self.task == 'car':
            low = compress(pair, quality_factor=self.quality_factor)
            low = self._np2Tensor(low, rgb_range=self.args.rgb_range)
            pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
        
        low = torch.clamp(low, 0, self.args.rgb_range)
        pair_t = torch.clamp(pair_t, 0, self.args.rgb_range)

        return low, pair_t, filename

    def get_patch(self, hr, scale=0):
        if scale == 0:
            scale = self.scale[self.idx_scale]
        hr = get_patch_img(hr, patch_size=self.args.patch_size, scale=scale)
        if self.train and not self.args.no_augment: hr = self._augment(hr)
        return hr

    def _augment(self, lr, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip: lr = lr[:, ::-1, :]
        if vflip: lr = lr[::-1, :, :]
        if rot90: lr = lr.transpose(1, 0, 2)
        return lr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale        