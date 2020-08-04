#!/usr/bin/env bash

mode=1 # 0
gpu=0,1,2,3
dataset1='cifar-10' # 'mnist', 'imagenet'
dataset2='cifar-100' # 'mnist', 'imagenet'
checkpoint='checkpoint_99.pth.tar'
batch=64
epoch=100
model='resnet18' # 'mlp', 'resnet18', 'resnet50', 'resnet101'
optimizer='SGD' # 'Adam'
lr=1e-3
dropout=0
bn_momentum=1e-2
weight_decay=1e-5
momentum=0
threshold=1e-4
workers=16
seed=0

# if you want to resume checkpoint, use "--resume"

if [ ${mode} == 0 ]; then
    # echo "(Baseline)"

    python main.py --gpu ${gpu} --dataset1 ${dataset1} --dataset2 ${dataset2} \
    --checkpoint ${checkpoint} --batch ${batch} --epoch ${epoch} \
    --model ${model} --optimizer ${optimizer} --lr ${lr} --dropout ${dropout} --bn_momentum ${bn_momentum} \
    --weight_decay ${weight_decay} --momentum ${momentum} --threshold ${threshold} --workers ${workers} --verbose --seed ${seed}
    
    python main.py --eval --gpu ${gpu} --dataset1 ${dataset1} --dataset2 ${dataset2} \
    --save ${save} --checkpoint ${checkpoint} --resume ${resume} --batch ${batch} --epoch ${epoch} \
    --model ${model} --optimizer ${optimizer} --lr ${lr} --dropout ${dropout} --bn_momentum ${bn_momentum} \
    --weight_decay ${weight_decay} --momentum ${momentum} --workers ${workers} --verbose --seed ${seed}

elif [ ${mode} == 1 ]; then
    # echo "(Costout)"

    python main.py --costout --gpu ${gpu} --dataset1 ${dataset1} --dataset2 ${dataset2} \
    --checkpoint ${checkpoint} --batch ${batch} --epoch ${epoch} \
    --model ${model} --optimizer ${optimizer} --lr ${lr} --dropout ${dropout} --bn_momentum ${bn_momentum} \
    --weight_decay ${weight_decay} --momentum ${momentum} --threshold ${threshold} --workers ${workers} --verbose --seed ${seed}

    python main.py --eval --costout --gpu ${gpu} --dataset1 ${dataset1} --dataset2 ${dataset2} \
    --checkpoint ${checkpoint} --resume ${resume} --batch ${batch} --epoch ${epoch} \
    --model ${model} --optimizer ${optimizer} --lr ${lr} --dropout ${dropout} --bn_momentum ${bn_momentum} \
    --weight_decay ${weight_decay} --momentum ${momentum} --workers ${workers} --verbose --seed ${seed}
fi

# python main.py --gpu 0,1,2,3