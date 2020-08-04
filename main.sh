#!/usr/bin/env bash

mode=1 # 0
gpu=0,1,2,3
cfg='configs/base.yaml'
# use costout, "--costout"
# evaluation mode, "--eval"
# resume checkpoint, "--resume"

if [ ${mode} == 0 ]; then # baseline
    python main.py --gpu ${gpu} --cfg ${cfg}

    python main.py --gpu ${gpu} --eval --resume ${resume} --cfg ${cfg}

elif [ ${mode} == 1 ]; then # costout
    python main.py --gpu ${gpu} --costout --cfg ${cfg}

    python main.py --gpu ${gpu} --costout --eval --resume ${resume} --cfg ${cfg}
fi

# python main.py --gpu 0,1,2,3 --coustout