#!/bin/bash

set -e

common_args="--exp_index 1 --batch_size 512 --lr 1e-4 --epochs 100 --pretrained True --dataset autovit-multi --dataset_path ../autovit_dataset/ --wandb_log True --wandb_group autovit-class-percent --predict_car_model"
debug_args="--debug --exp_index 1 --batch_size 512 --lr 1e-4 --epochs 1 --pretrained True --dataset autovit-multi --dataset_path ../autovit_dataset/ --wandb_group autovit-class-percent --predict_car_model"

# Unified with all data percentages from 1 to 0.1
python main.py $common_args --model vit-small-unified --trainer classic --mixup --run_name autovit-vit-unified-1

python main.py $common_args --model vit-small-unified --trainer classic --dataset_percent 0.5 --mixup --run_name autovit-vit-unified-0.5

python main.py $common_args --model vit-small-unified --trainer classic --dataset_percent 0.2 --mixup --run_name autovit-vit-unified-0.2

python main.py $common_args --model vit-small-unified --trainer classic --dataset_percent 0.1 --mixup --run_name autovit-vit-unified-0.1

# Separate with all data percentages from 1 to 0.1
python main.py $common_args --model vit-small-separate --trainer multitask --mixup --predict_full --run_name autovit-vit-separate-1

python main.py $common_args --model vit-small-separate --trainer multitask --dataset_percent 0.5 --mixup --predict_full --run_name autovit-vit-separate-0.5

python main.py $common_args --model vit-small-separate --trainer multitask --dataset_percent 0.2 --mixup --predict_full --run_name autovit-vit-separate-0.2

python main.py $common_args --model vit-small-separate --trainer multitask --dataset_percent 0.1 --mixup --predict_full --run_name autovit-vit-separate-0.1

# Conditional with all data percentages from 1 to 0.1
python main.py $common_args --model vit-small-conditional --trainer multitask --mixup --predict_full --run_name autovit-vit-conditional-1

python main.py $common_args --model vit-small-conditional --trainer multitask --dataset_percent 0.5 --mixup --predict_full --run_name autovit-vit-conditional-0.5

python main.py $common_args --model vit-small-conditional --trainer multitask --dataset_percent 0.2 --mixup --predict_full --run_name autovit-vit-conditional-0.2

python main.py $common_args --model vit-small-conditional --trainer multitask --dataset_percent 0.1 --mixup --predict_full --run_name autovit-vit-conditional-0.1
