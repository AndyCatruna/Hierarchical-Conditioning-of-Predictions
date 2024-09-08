import argparse
import numpy as np
import torch
import os

def define_args():
    parser = argparse.ArgumentParser()

    # Learning Hyperparameters
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--scheduler', choices=['step', 'onecycle'], default='step')

    # Model Hyperparameters
    parser.add_argument('--model', type = str, default = 'swin-small-conditional')
    parser.add_argument('--pretrained', type = bool, default = False)
    
    # Trainer Hyperparameters
    parser.add_argument('--trainer', choices=['classic', 'multitask'], default='classic')
    parser.add_argument('--augmentation', choices=['no-aug', 'weak-aug', 'rand-aug', 'auto-aug'], default='rand-aug')
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--cutmix', action='store_true', default=False)
    parser.add_argument('--dataset', choices=['autovit-multi', 'stanford-multi', 'car_model_dataset-multi'], default='stanford')
    parser.add_argument('--loss_weight', type = str, choices=['static', 'soft_adapt', 'soft_adapt2'], default='static')

    # Misc Hyperparameters
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--wandb_group', type=str, default='experiments')
    parser.add_argument('--exp_index', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default = '')
    parser.add_argument('--run_name', type=str, default = '')
    parser.add_argument('--dataset_path', type=str, default = '../car_model_dataset/')
    parser.add_argument('--old_out_features', type=int, default = 1958)
    parser.add_argument('--seed', type=int, default = 42)

    # Task Hyperparameters
    parser.add_argument('--predict_car_model', action='store_true', default=False)
    parser.add_argument('--predict_angle', action='store_true', default=False)
    parser.add_argument('--predict_km', action='store_true', default=False)
    parser.add_argument('--predict_price', action='store_true', default=False)
    parser.add_argument('--predict_body', action='store_true', default=False)
    parser.add_argument('--predict_full', action='store_true', default=False)    

    # Gating Hyperparameters
    parser.add_argument('--use_residual', action='store_true', default=False)
    parser.add_argument('--use_consistency_loss', action='store_true', default=False)
    parser.add_argument('--gating_depth', type=int, default=2)
    parser.add_argument('--gating_num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Dataset hyperparameters
    parser.add_argument('--dataset_percent', type=float, default=1.0)

    # Pretrain hyperparameters
    parser.add_argument('--pretrain_transfer', action='store_true', default=False)
    parser.add_argument('--to_dataset', type=str, default='')
    parser.add_argument('--from_dataset', type=str, default='')
    parser.add_argument('--save_checkpoint', type=str, default='')
    parser.add_argument('--load_checkpoint', type=str, default='')
    

    # Debug hyperparameters
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    return args

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def load_checkpoint(model, weight_path):
    if os.path.exists(weight_path):
        print('Loading checkpoint from {}'.format(weight_path))
        model.load_state_dict(torch.load(weight_path))

def save_checkpoint(model, weight_path):
    print('Saving checkpoint to {}'.format(weight_path))
    torch.save(model.state_dict(), weight_path)