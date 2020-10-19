#!/usr/bin/env bash

mode='mt'
lr=1e-3
epoch=1000
batch=512
len=10
layers=5
dim=1024
seed=0

# python binary_predictor.py --mode ${mode} --lr ${lr} --epoch ${epoch} --len ${len} --seed ${seed}

# multi-mode
python binary_predictor.py --mode 'st' --lr 1e-1 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'st' --lr 1e-2 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'st' --lr 1e-3 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'st' --lr 1e-4 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'st' --lr 1e-5 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}

python binary_predictor.py --mode 'mt' --lr 1e-1 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'mt' --lr 1e-2 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'mt' --lr 1e-3 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'mt' --lr 1e-4 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}
python binary_predictor.py --mode 'mt' --lr 1e-5 --epoch ${epoch} --batch ${batch} --len 10 --layers ${layers} --dim ${dim} --seed ${seed}

# python binary_predictor.py --mode 'st' --lr 1e-1 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-2 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-3 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-4 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'st' --lr 1e-5 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}

# python binary_predictor.py --mode 'mt' --lr 1e-1 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-2 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-3 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-4 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}
# python binary_predictor.py --mode 'mt' --lr 1e-5 --epoch 100 --batch ${batch} --len 20 --layers ${layers} --dim ${dim} --seed ${seed}