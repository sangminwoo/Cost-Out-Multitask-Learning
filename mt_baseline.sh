#!/usr/bin/env bash

gpu=0,1,2,3
mode='mt_baseline'
cfg='configs/baseline_base.yaml'

python main.py --gpu ${gpu} --mode ${mode} --cfg ${cfg}
python main.py --gpu ${gpu} --mode ${mode} --cfg ${cfg} --eval --resume 

# python main.py --gpu 0,1,2,3 --coustout