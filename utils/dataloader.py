import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class ImageLoader:
	def __init__(self, args):
		self.args = args

	def get_dataset(self, dataset, phase):
		assert dataset in ['imagenet', 'cifar10', 'cifar100']
		assert phase in ['train', 'val', 'test']
		if dataset == 'imagenet': # 1.4Mx?x?
			MEAN = [0.485, 0.456, 0.406]
			STD = [0.229, 0.224, 0.225]

			if phase == 'train':
				transform = transforms.Compose([
									transforms.RandomResizedCrop(256),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])
			else:
				transform = transforms.Compose([
										transforms.Resize(256),
										transforms.CenterCrop(256),
										transforms.ToTensor(),
										transforms.Normalize(mean=MEAN, std=STD)])

			dataset = datasets.CIFAR100(self.args.root_dir, train=False if phase=='test' else True,
										transform=transform, download=True)

		elif dataset == 'cifar10': # 60000x32x32
			MEAN = [0.4914, 0.4822, 0.4465]
			STD = [0.2023, 0.1994, 0.2010]

			if phase == 'train':
				transform = transforms.Compose([
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])
			else:
				transform = transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize(mean=MEAN, std=STD)])

			dataset = datasets.CIFAR100(self.args.root_dir, train=False if phase=='test' else True,
										transform=transforms.Nor, download=True)

		elif dataset == 'cifar100': # 60000x32x32
			MEAN = [0.5071, 0.4867, 0.4408]
			STD = [0.2675, 0.2565, 0.2761]

			if phase == 'train':
				transform = transforms.Compose([
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])
			else:
				transform = transforms.Compose([
										transforms.ToTensor(),
										transforms.Normalize(mean=MEAN, std=STD)])

			dataset = datasets.CIFAR100(self.args.root_dir, train=False if phase=='test' else True,
										transform=transform, download=True)

		return dataset

	def get_train_loader(self, shuffle=True, valid_size=0.1):
		train_dataset = self.get_dataset(phase='train')
		valid_dataset = self.get_dataset(phase='val')

		num_train = len(train_dataset)
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))

		if shuffle:
			np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)

		train_loader = DataLoader(train_dataset, batch_size=self.args.batch, sampler=train_sampler,
			num_workers=self.args.workers, pin_memory=True)

		valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch, sampler=valid_sampler,
			num_workers=self.args.workers, pin_memory=True)

		return train_loader, valid_loader

	def get_test_loader(self):
		test_dataset = self.get_dataset(phase='test')

		test_loader = DataLoader(test_dataset, batch_size=self.args.batch, shuffle=False,
        	num_workers=self.args.workers, pin_memory=True)

		return test_loader