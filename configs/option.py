import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rich CNN-Transformer Feature Aggregation Networks for SR')

    parser.add_argument('--release', action='store_true', 
                        help='store true for inference using pretrained weights')

    # Hardware configurations
    parser.add_argument('--num_workers', type=int, default=16, 
                        help='number of threads for data loading')
    parser.add_argument('--cpu', action='store_true', 
                        help='use cpu only')
    parser.add_argument('--gpus', type=int, default=1, 
                        help='number of GPUs')
    parser.add_argument('--seed', type=int, default=310, 
                        help='random seed')

    parser.add_argument('--task', default='sr', 
                        help='sr or car')

    # Model specifications
    parser.add_argument('--model', default='ACT', 
                        help='name of model')
    parser.add_argument('--n_feats', type=int, default=64, 
                        help='number of feature maps')
    ## CNN branch
    parser.add_argument('--act', type=str, default='relu', 
                        help='activation function')
    parser.add_argument('--n_resgroups', type=int, default=4, 
                        help='number of residual groups')
    parser.add_argument('--n_resblocks', type=int, default=12, 
                        help='number of residual blocks')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    ## Transformer branch
    parser.add_argument('--token_size', type=int, default=3, 
                        help='size of token')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of haeds for multi-head self-attention')
    parser.add_argument('--n_layers', type=int, default=8, 
                        help='number of transformer blocks')
    parser.add_argument('--dropout_rate', type=float, default=0, 
                        help='dropout rate for mlp block')
    parser.add_argument('--expansion_ratio', type=int, default=4, 
                        help='expansion ratio for mlp block')
    ## Fusion block
    parser.add_argument('--n_fusionblocks', type=int, default=4, 
                        help='number of fusion blocks')

    # Training specifications
    parser.add_argument('--epochs', type=int, default=150, 
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='input batch size for training')

    # Test specifications
    parser.add_argument('--self_ensemble', action='store_true', 
                        help='use self-ensemble method for test')
    parser.add_argument('--crop_batch_size', type=int, default=64, 
                        help='input batch size for testing')
    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='learning rate')
    parser.add_argument('--decay', type=int, default=50, 
                        help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor for step decay')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), 
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0, 
                        help='weight decay')

    # Log specifications
    parser.add_argument('--save_path', type=str, default=None, 
                        help='path to save')
    parser.add_argument('--ckpt_path', type=str, default=None, 
                        help='path to checkpoint')
    parser.add_argument('--num_sanity_val_steps', type=int, default=2, 
                        help='sanity val steps')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='control epoch size')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='validation frequency')
    parser.add_argument('--precision', type=int, default=32, 
                        help='16 or 32')

    # Data configurations
    parser.add_argument('--dir_data', type=str, default='./datasets', 
                        help='dataset directory')
    parser.add_argument('--data_train', type=str, default='ImageNet', 
                        help='train dataset name')
    parser.add_argument('--data_test', type=str, default='Set14', 
                        help='test dataset name')
    parser.add_argument('--ext', type=str, default='img', 
                        help='dataset file extension')
    parser.add_argument('--scale', type=int, default=1, 
                        help='super resolution scale')
    parser.add_argument('--quality_factor', type=int, default=40,
                        help='quality factor for image compression')
    parser.add_argument('--patch_size', type=int, default=48, 
                        help='input patch size')
    parser.add_argument('--rgb_range', type=int, default=255, 
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3, 
                        help='number of color channels')
    parser.add_argument('--no_augment', action='store_true', 
                        help='whether to use data augmentation')

    args = parser.parse_args()

    return args
