import os
import datetime
from os import path as osp

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from src.models import create_model
from src.data import create_datamodule
from configs.option import parse_args


def main():
    # parse arguments
    args = parse_args()

    # make directory to save experiment
    save_dir = 'experiments/test'
    if not args.save_path:
        # generate random name if not given
        now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        args.save_path = osp.join(save_dir, now)
    else:
        args.save_path = osp.join(save_dir, args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    print(f'Experimental results will be saved at: {args.save_path}')

    # fix random seed
    seed_everything(args.seed)

    # create model
    model = create_model(args)

    # create datamodule
    datamodule = create_datamodule(args)

    # define trainer
    trainer = Trainer(
        default_root_dir=args.save_path, gpus=1, enable_progress_bar=False
    )

    # test model
    print('Start test!')
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
