import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_dataset(root, dataset, phase):
	assert dataset in ['mnist', 'cifar10', 'cifar100', 'imagenet']
	assert phase in ['train', 'val', 'test']
	
	if dataset == 'mnist': # 60000x28x28
		# MEAN = 0.1307
		# STD = 0.3081

		transform = transforms.Compose([
							transforms.ToTensor()])
							# transforms.Normalize(mean=MEAN, std=STD)])

		dataset = datasets.MNIST(root=root, train=False if phase=='test' else True,
								 transform=transform, download=True if not os.path.exists(root) else False)

	elif dataset == 'cifar10': # 60000x32x32
		MEAN = [0.4914, 0.4822, 0.4465]
		STD = [0.2023, 0.1994, 0.2010]

		if phase == 'train':
			transform = transforms.Compose([
							transforms.RandomResizedCrop(224),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(mean=MEAN, std=STD)])
		else:
			transform = transforms.Compose([
							transforms.Resize(224),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(mean=MEAN, std=STD)])

		dataset = datasets.CIFAR10(root=root, train=False if phase=='test' else True,
								   transform=transform, download=True if not os.path.exists(root) else False)

	elif dataset == 'cifar100': # 60000x32x32
		MEAN = [0.5071, 0.4867, 0.4408]
		STD = [0.2675, 0.2565, 0.2761]

		if phase == 'train':
			transform = transforms.Compose([
							transforms.RandomResizedCrop(224),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(mean=MEAN, std=STD)])
		else:
			transform = transforms.Compose([
							transforms.Resize(224),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(mean=MEAN, std=STD)])

		dataset = datasets.CIFAR100(root=root, train=False if phase=='test' else True,
									transform=transform, download=True if not os.path.exists(root) else False)

	elif dataset == 'imagenet': # 1.4Mx?x?
		MEAN = [0.485, 0.456, 0.406]
		STD = [0.229, 0.224, 0.225]

		if phase == 'train':
			transform = transforms.Compose([
							transforms.RandomResizedCrop(224),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(mean=MEAN, std=STD)])
		else:
			transform = transforms.Compose([
							transforms.Resize(224),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(mean=MEAN, std=STD)])

		dataset = datasets.ImageNet(root=root, train=False if phase=='test' else True,
									transform=transform, download=True if not os.path.exists(root) else False)

	return dataset