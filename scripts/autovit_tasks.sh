#!/bin/bash

set -e

common_args="--exp_index 1 --batch_size 512 --lr 1e-4 --epochs 200 --pretrained True --dataset autovit-multi --dataset_path ../autovit_dataset/ --wandb_log True --wandb_group autovit-tasks --model swin-small-conditional --trainer multitask --mixup"
debug_args="--debug --exp_index 1 --batch_size 512 --lr 1e-4 --epochs 1 --pretrained True --dataset autovit-multi --dataset_path ../autovit_dataset/ --wandb_group autovit-tasks --model swin-small-conditional --trainer multitask --mixup"

### Single tasks
python main.py $common_args --predict_car_model --predict_full --run_name autovit-single-model
python main.py $common_args --predict_angle --run_name autovit-single-angle
python main.py $common_args --predict_km --run_name autovit-single-km
python main.py $common_args --predict_price --run_name autovit-single-price


### Double tasks
# With car model
python main.py $common_args --predict_car_model --predict_full --predict_angle --run_name autovit-double-model-angle
python main.py $common_args --predict_car_model --predict_full --predict_km --run_name autovit-double-model-km
python main.py $common_args --predict_car_model --predict_full --predict_price --run_name autovit-double-model-price

# With angle
python main.py $common_args --predict_angle --predict_km --run_name autovit-double-angle-km
python main.py $common_args --predict_angle --predict_price --run_name autovit-double-angle-price

# With km
python main.py $common_args --predict_km --predict_price --run_name autovit-double-km-price


### Triple tasks
python main.py $common_args --predict_car_model --predict_full --predict_angle --predict_km --run_name autovit-triple-model-angle-km
python main.py $common_args --predict_car_model --predict_full --predict_angle --predict_price --run_name autovit-triple-model-angle-price
python main.py $common_args --predict_car_model --predict_full --predict_km --predict_price --run_name autovit-triple-model-km-price
python main.py $common_args --predict_angle --predict_km --predict_price --run_name autovit-triple-angle-km-price


### All tasks
python main.py $common_args --predict_car_model --predict_full --predict_angle --predict_km --predict_price --run_name autovit-all-model-angle-km-price



