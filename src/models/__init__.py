import requests
from os import path as osp

import torch

from .act_net import ACT
from .act_module import ACTLitModule


def create_model(args, is_train=False):
    if is_train:
        return ACTLitModule(net=ACT(args), args=args)

    else: # test setting
        if args.release:
            model_path = f'pretrained_weights/act_{args.task}_x{args.scale}.pt'
            if not osp.exists(model_path):
                # download pretrained weight
                url = f'https://github.com/jinsuyoo/ACT/releases/download/v0.0.0/{osp.basename(model_path)}'
                r = requests.get(url, allow_redirects=True)
                print(f'Downloading pretrained weight: {model_path}')
            
            net = ACT(args)
            net.load_state_dict(torch.load(model_path))

            return ACTLitModule(net=net, args=args)

        elif args.ckpt_path is not None:
            # use pretrained parameter
            assert osp.exists(args.ckpt_path), print(f'checkpoint not exists: {args.ckpt_path}')
            print(f'Loading checkpoint from: {args.ckpt_path}')

            return ACTLitModule.load_from_checkpoint(args.ckpt_path, args=args)
        
        else:
            raise ValueError('Need release option or checkpoint path')