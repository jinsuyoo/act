import os
from os import path as osp
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from einops import rearrange

from src.utils.utils_image import quantize, calc_psnr


class ACTLitModule(LightningModule):
    def __init__(self, net: nn.Module, args: Namespace):
        super().__init__()

        self.args = args
        
        self.net = net

        self.task = args.task

        self.rgb_range = args.rgb_range
        self.scale = args.scale
        self.crop_batch_size = args.crop_batch_size 
        self.patch_size = args.patch_size
        self.self_ensemble = args.self_ensemble

        # optimization configs
        self.lr = args.lr
        self.step_size = args.decay
        self.gamma = args.gamma
        self.betas = args.betas
        self.eps = args.epsilon
        self.weight_decay = args.weight_decay

        self.save_path = args.save_path

        self.criterion = nn.L1Loss()

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f'Skip loading parameter: {k}, '
                                f'required shape: {model_state_dict[k].shape}, '
                                f'loaded shape: {state_dict[k].shape}')
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f'Dropping parameter {k}')
                is_changed = True

        if is_changed:
            checkpoint.pop('optimizer_states', None)

    def forward(self, x):
        return self.net(x)

    def forward_x8(self, x, forward_function):
        # this code is borrowed from https://github.com/yulunzhang/RCAN
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).cuda()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def forward_chop(self, x, shave=12):
        # this code is borrowed from https://github.com/huawei-noah/Pretrained-IPT with few modification
        
        x = x.cpu()
        b, _, h, w = x.shape

        scale = self.scale
        pad_size = self.patch_size
        shave = self.patch_size // 2

        h_cut = (h - pad_size) % (int(shave / 2))
        w_cut = (w - pad_size) % (int(shave / 2))

        x_unfold = F.unfold(x, pad_size, stride=int(shave / 2))
        x_unfold = rearrange(
            x_unfold, 'b (c p1 p2) n -> n (b c) p1 p2', p1=pad_size, p2=pad_size
        )
        x_unfold = x_unfold.cuda()

        n = x_unfold.shape[0]

        y_unfold = [
            self(x_unfold[i * self.crop_batch_size : (i + 1) * self.crop_batch_size, ...]).cpu()
            for i in range(n // self.crop_batch_size + (n % self.crop_batch_size != 0))
        ]

        y_unfold = torch.cat(y_unfold, dim=0)
        y = F.fold(
            rearrange(y_unfold, 'n (b c) h w -> b (c h w) n', b=b),
            output_size=((h - h_cut) * scale, (w - w_cut) * scale),
            kernel_size=pad_size * scale,
            stride=int(shave / 2 * scale),
        )

        x_hw_cut = x[..., (h - pad_size) :, (w - pad_size) :]
        y_hw_cut = self(x_hw_cut.cuda()).cpu()

        x_h_cut = x[..., (h - pad_size) :, :]
        x_w_cut = x[..., :, (w - pad_size) :]
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, pad_size, shave)
        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, pad_size, shave)

        x_h_top = x[..., :pad_size, :]
        x_w_top = x[..., :, :pad_size]
        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, pad_size, shave)
        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, pad_size, shave)

        y[..., : pad_size * scale, :] = y_h_top
        y[..., :, : pad_size * scale] = y_w_top

        y_unfold = y_unfold[
            ...,
            int(shave / 2 * scale) : pad_size * scale - int(shave / 2 * scale),
            int(shave / 2 * scale) : pad_size * scale - int(shave / 2 * scale),
        ].contiguous()

        y_inter = F.fold(
            y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            output_size=((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
            kernel_size=(pad_size - shave) * scale,
            stride=int(shave / 2 * scale),
        )

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = F.fold(
            F.unfold(
                y_ones, pad_size * scale - shave * scale, stride=int(shave / 2 * scale)
            ),
            output_size=((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
            kernel_size=(pad_size - shave) * scale,
            stride=int(shave / 2 * scale),
        )

        y_inter = y_inter / divisor

        y[..., int(shave / 2 * scale) : (h - h_cut) * scale - int(shave / 2 * scale), 
               int(shave / 2 * scale) : (w - w_cut) * scale - int(shave / 2 * scale),
        ] = y_inter

        y = torch.cat(
            [y[..., : y.size(2) - int((pad_size - h_cut) / 2 * scale), :],
             y_h_cut[..., int((pad_size - h_cut) / 2 * scale + 0.5) :, :]
            ], dim=2
        )
        y_w_cat = torch.cat(
            [y_w_cut[..., : y_w_cut.size(2) - int((pad_size - h_cut) / 2 * scale), :],
             y_hw_cut[..., int((pad_size - h_cut) / 2 * scale + 0.5) :, :]
            ], dim=2
        )
        y = torch.cat(
            [y[..., :, : y.size(3) - int((pad_size - w_cut) / 2 * scale)],
             y_w_cat[..., :, int((pad_size - w_cut) / 2 * scale + 0.5) :]
            ], dim=3
        )
        return y.cuda()

    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, pad_size, shave):
        x_h_cut_unfold = F.unfold(x_h_cut, pad_size, stride=int(shave / 2))
        b, d, n = x_h_cut_unfold.shape
        x_h_cut_unfold = rearrange(
            x_h_cut_unfold, 'b (c ph pw) n -> n (b c) ph pw', ph=pad_size, pw=pad_size
        )

        x_range = n // self.crop_batch_size + (n % self.crop_batch_size != 0)

        y_h_cut_unfold = []
        x_h_cut_unfold = x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(
                self(
                    x_h_cut_unfold[
                        i * self.crop_batch_size : (i + 1) * self.crop_batch_size, ...
                    ]
                ).cpu()
            )

        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = F.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            (pad_size * self.scale, (w - w_cut) * self.scale),
            pad_size * self.scale,
            stride=int(shave / 2 * self.scale),
        )

        y_h_cut_unfold = y_h_cut_unfold[
            ...,
            :,
            int(shave / 2 * self.scale) : pad_size * self.scale
            - int(shave / 2 * self.scale),
        ].contiguous()

        y_h_cut_inter = F.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            (pad_size * self.scale, (w - w_cut - shave) * self.scale),
            (pad_size * self.scale, pad_size * self.scale - shave * self.scale),
            stride=int(shave / 2 * self.scale),
        )

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)

        divisor = F.fold(
            F.unfold(
                y_ones,
                (pad_size * self.scale, pad_size * self.scale - shave * self.scale),
                stride=int(shave / 2 * self.scale)
            ),
            (pad_size * self.scale, (w - w_cut - shave) * self.scale),
            (pad_size * self.scale, pad_size * self.scale - shave * self.scale),
            stride=int(shave / 2 * self.scale)
        )
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[
            ...,
            :,
            int(shave / 2 * self.scale) : (w - w_cut) * self.scale - int(shave / 2 * self.scale),
        ] = y_h_cut_inter

        return y_h_cut

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, pad_size, shave):
        x_w_cut_unfold = F.unfold(x_w_cut, pad_size, stride=int(shave / 2))
        b, d, n = x_w_cut_unfold.shape
        x_w_cut_unfold = rearrange(
            x_w_cut_unfold, 'b (c ph pw) n -> n (b c) ph pw', ph=pad_size, pw=pad_size
        )

        x_range = n // self.crop_batch_size + (n % self.crop_batch_size != 0)

        y_w_cut_unfold = []
        x_w_cut_unfold = x_w_cut_unfold.cuda()
        for i in range(x_range):
            y_w_cut_unfold.append(
                self(
                    x_w_cut_unfold[
                        i * self.crop_batch_size : (i + 1) * self.crop_batch_size, ...
                    ]
                ).cpu()
            )

        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = F.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            ((h - h_cut) * self.scale, pad_size * self.scale),
            pad_size * self.scale,
            stride=int(shave / 2 * self.scale),
        )

        y_w_cut_unfold = y_w_cut_unfold[
            ...,
            int(shave / 2 * self.scale) : pad_size * self.scale
            - int(shave / 2 * self.scale),
            :,
        ].contiguous()

        y_w_cut_inter = F.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            ((h - h_cut - shave) * self.scale, pad_size * self.scale),
            (pad_size * self.scale - shave * self.scale, pad_size * self.scale),
            stride=int(shave / 2 * self.scale),
        )

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)

        divisor = F.fold(
            F.unfold(
                y_ones,
                (pad_size * self.scale - shave * self.scale, pad_size * self.scale),
                stride=int(shave / 2 * self.scale)
            ),
            ((h - h_cut - shave) * self.scale, pad_size * self.scale),
            (pad_size * self.scale - shave * self.scale, pad_size * self.scale),
            stride=int(shave / 2 * self.scale)
        )
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[
            ...,
            int(shave / 2 * self.scale) : (h - h_cut) * self.scale - int(shave / 2 * self.scale), 
            :
        ] = y_w_cut_inter

        return y_w_cut

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        img_lq, img_gt, _ = train_batch
        
        output = self(img_lq)

        loss = self.criterion(output, img_gt)

        self.log('train/loss', loss, 
                 prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        img_lq, img_gt, filename = val_batch
        
        output = self.forward_chop(img_lq)        
        
        loss = self.criterion(output, img_gt).detach()

        output = quantize(output, self.rgb_range)

        psnr = calc_psnr(output, img_gt, self.scale, self.rgb_range)

        self.log('val/loss', loss, 
                 prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/psnr', psnr, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # validation images for tensorboard visualization
        val_img_name = 'val/val_image_{}'.format(filename[0].split('/')[-1])
        grid = make_grid(output / 255)
        self.logger.experiment.add_image(val_img_name, grid, self.trainer.current_epoch)

    def on_test_start(self):
        from src.utils.utils_logger import get_logger
        from src.utils.utils_saver import Saver

        assert self.task in ['sr', 'car']

        self.data_test = self.args.data_test

        self.save_dir_images = osp.join(
            self.save_path, 'images', 'results-{}'.format(self.data_test)
        )
        if not osp.exists(self.save_dir_images):
            os.makedirs(self.save_dir_images, exist_ok=True)

        self.saver = Saver()
        self.saver.begin_background()

        self.text_logger = get_logger(log_path=osp.join(self.save_path, 'result.log'))

        self.text_logger.info(f'Test dataset: {self.data_test}')
        self.text_logger.info(f'Scale factor: {self.scale}')

        self.border = self.scale if self.task == 'sr' else 0

        self.avg_psnr = []

    def test_step(self, batch, batch_idx):
        img_lq, img_gt, filename = batch

        if self.self_ensemble:
            # x8 self-ensemble
            output = self.forward_x8(img_lq, self.forward_chop)
        else:
            output = self.forward_chop(img_lq)

        output = quantize(output, self.rgb_range)

        psnr = calc_psnr(output, img_gt, self.scale, self.rgb_range)

        self.text_logger.info(f'Filename: {filename[0]} | PSNR: {psnr:.3f}')
        self.avg_psnr.append(psnr)

        self.saver.save_results(
            save_dir=self.save_dir_images,
            filename=filename[0],
            save_list=[output, img_lq, img_gt],
            scale=self.scale,
            rgb_range=self.rgb_range
        )

    def on_test_end(self):
        avg_psnr = np.mean(np.array(self.avg_psnr))
        self.text_logger.info(f'Average PSNR: {avg_psnr:.3f}')
        self.saver.end_background()
