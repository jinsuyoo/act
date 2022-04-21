import os
import datetime
from os import path as osp

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import create_model
from src.data import create_datamodule
from configs.option import parse_args


def main():
    args = parse_args()

    # make directory to save experiment
    save_dir = 'experiments/train'
    if not args.save_path:
        now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        args.save_path = osp.join(save_dir, now) 
    else: 
        args.save_path = osp.join(save_dir, args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    print(f'Experimental results will be saved at: {args.save_path}')

    # fix random seed
    seed_everything(args.seed)
    
    # define logger object
    logger = TensorBoardLogger(osp.join(args.save_path, 'tb_logs'), name='act')

    # create model
    model = create_model(args, is_train=True)

    # create datamodule
    datamodule = create_datamodule(args)

    # specify checkpoint configs
    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(args.save_path, 'checkpoint'),
        filename='best-epoch{epoch:02d}-val_psnr{val/psnr:.2f}',
        monitor='val/psnr',
        save_last=True,
        save_top_k=1,
        mode='max',
        auto_insert_metric_name=False
    )
    
    # define lightning trainer
    trainer = Trainer(
        strategy='ddp',
        accelerator='gpu', 
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        default_root_dir=args.save_path,
        max_epochs=args.epochs, 
        limit_train_batches=args.limit_train_batches,
        val_check_interval=args.val_check_interval,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_val_steps,
        resume_from_checkpoint=None
    )

    # begin training
    trainer.fit(model=model, datamodule=datamodule)
        

if __name__ == '__main__':
    main()