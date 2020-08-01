#!/usr/bin/env bash

mode=0 # 1
gpu=0,1,2,3

if [ ${mode} == 0 ]; then
    echo "(Baseline)"
    python main.py --gpu ${gpu}
    python main.py --gpu ${gpu} --eval
elif [ ${mode} == 1 ]; then
    echo "(Costout))"
    python main.py --gpu ${gpu} --costout
    python main.py --gpu ${gpu} --eval --costout
fi