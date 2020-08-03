import numpy as np
from torchvision import datasets

def calc_mean_std(dataset='imagenet'):
	if dataset == 'imagenet':
		train_data = datasets.ImageNet('./imagenet', train=True, download=True)
	elif dataset == 'cifar10':
		train_data = datasets.CIFAR10('./cifar10', train=True, download=True)
	elif dataset == 'cifar100'
		train_data = datasets.CIFAR100('./cifar100', train=True, download=True)

	x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
	train_mean = np.mean(x, axis=(0, 1))
	train_std = np.std(x, axis=(0, 1))
	print(train_mean/255., train_std/255.)

if __name__ == '__main__':
	calc_mean_std(dataset='cifar10')