#!/usr/bin/env bash

mode=0
gpu=0,1,2,3
cfg='configs/base.yaml'
# use costout, "--costout"
# evaluation mode, "--eval"
# resume checkpoint, "--resume"

if [ ${mode} == 0 ]; then # baseline
    python main.py --gpu ${gpu} --cfg ${cfg}

    python main.py --gpu ${gpu} --cfg ${cfg} --eval --resume 

elif [ ${mode} == 1 ]; then # costout
    python main.py --gpu ${gpu} --cfg ${cfg} --costout

    python main.py --gpu ${gpu} --cfg ${cfg} --costout --eval --resume 
fi

# python main.py --gpu 0,1,2,3 --coustout