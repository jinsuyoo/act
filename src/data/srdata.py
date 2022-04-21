import os
import os.path as osp
import glob
import random
import pickle

import imageio
import torch.utils.data as data

from src.data import common


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = args.model == 'VDSR'
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = osp.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()

        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
            if self.scale == 1:
                os.makedirs(osp.join(self.dir_hr), exist_ok=True)
            else:
                os.makedirs(
                    osp.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(self.scale),
                    ),
                    exist_ok=True,
                )

            self.images_hr, self.images_lr = [], []
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for l in list_lr:
                b = l.replace(self.apath, path_bin)
                b = b.replace(self.ext[1], '.pt')
                self.images_lr.append(b)
                self._check_and_load(args.ext, l, b, verbose=True)
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(glob.glob(osp.join(self.dir_hr, '*' + self.ext[0])))
        names_lr = []
        for f in names_hr:
            filename, _ = osp.splitext(osp.basename(f))
            if self.scale != 1:
                names_lr.append(
                    osp.join(
                        self.dir_lr,
                        'X{}/{}x{}{}'.format(
                            self.scale, filename, self.scale, self.ext[1]
                        ),
                    )
                )
        if self.scale == 1:
            names_lr = names_hr

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = osp.join(dir_data, self.name)
        self.dir_hr = osp.join(self.apath, 'HR')
        self.dir_lr = osp.join(self.apath, 'LR_bicubic')
        # if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not osp.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file_hr(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = osp.splitext(osp.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)

        return hr, filename

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = osp.splitext(osp.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch_hr(self, hr):
        if self.train:
            hr = self.get_patch_img_hr(hr, patch_size=self.args.patch_size, scale=1)

        return hr

    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        ih, iw = img.shape[:2]

        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy : iy + ip, ix : ix + ip, :]

        return ret

    def get_patch(self, lr, hr):
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size=self.args.patch_size * self.scale, scale=self.scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0 : ih * self.scale, 0 : iw * self.scale]

        return lr, hr
