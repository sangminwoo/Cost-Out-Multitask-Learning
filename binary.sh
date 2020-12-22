#!/usr/bin/env bash

mode='st'
p_lossdrop=0.1
lr=1e-3
epoch=10000
batch=512
len=10
layers=3
in_dim=16 # 64
hid_dim=1024
seed=0

# singletask baseline
# python binary_predictor.py --mode 'st' --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# multitask baseline
# python binary_predictor.py --mode 'mt' --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# per-task filter
# python binary_predictor.py --mode 'mt' --filter --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# loss-dropout
# python binary_predictor.py --mode 'mt' --lossdrop --p_lossdrop ${p_lossdrop} --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# residual
# python binary_predictor.py --mode 'mt' --residual --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# adaptive loss balancing
python binary_predictor.py --mode 'mt' --adploss --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# filter + loss-drop
# python binary_predictor.py --mode 'mt' --filter --lossdrop --p_lossdrop ${p_lossdrop} --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# filter + loss-drop + residual
# python binary_predictor.py --mode 'mt' --filter --lossdrop --residual --p_lossdrop ${p_lossdrop} --lr ${lr} --epoch ${epoch} --batch ${batch} --len ${len} --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# multi-mode
# python binary_predictor.py --mode 'st' --lr 1e-1 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-2 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-3 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-4 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-5 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# python binary_predictor.py --mode 'mt' --lr 1e-1 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-2 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-3 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-4 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-5 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# python binary_predictor.py --mode 'st' --lr 1e-1 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-2 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-3 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-4 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-5 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}

# python binary_predictor.py --mode 'mt' --lr 1e-1 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-2 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-3 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-4 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-5 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --in_dim ${in_dim} --hid_dim ${hid_dim} --seed ${seed}