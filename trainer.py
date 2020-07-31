import os
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# Model
from mlp_multitask import MLP, build_mlp
from resnet_multitask import ResNet, build_resnet
# Dataset / DataLoader
from dataset import get_dataset
import ImageLoader, get_dataloader
# Validation
from utils import AverageMeter, accuracy

class Trainer:
	def __init__(self, args, epochs=100, model='resnet18', optimizer='SGD',	verbose=True):
	
		assert model in ['mlp', 'resnet18', 'resnet50', 'resnet101'], 'model not available!'
		assert optimizer in ['SGD', 'Adam'], 'optimizer not available!'

		self.args = args
		self.epochs = epochs
		self.dropout = args.dropout
		self.device = args.device
		self.save = args.save
		self.verbose = args.verbose
		self.costout = args.costout

		# Model
		if model == 'mlp':
			self.model = build_mlp(10, 512, 128, args.bn_momentum, args.dropout)
		elif model == 'resnet18':
			self.model = build_resnet(arch='resnet18', pretrained=False)
		elif model == 'resnet50':
			self.model = build_resnet(arch='resnet50', pretrained=False)
		elif model == 'resnet101':
			self.model = build_resnet(arch='resnet101', pretrained=False)
		self.fc1 = nn.Linear(128, 10)
		self.fc2 = nn.Linear(128, 100)

		self.model = nn.DataParallel(self.model.to(self.device))
		self.fc1 = nn.DataParallel(self.fc1.to(self.device))
		self.fc2 = nn.DataParallel(self.fc2.to(self.device))

		self.criterion = nn.CrossEntropyLoss().to(self.device)
		if optimizer == 'SGD':
			self.optimizer = optim.SGD(	self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		# Dataset / DataLoader
		if args.dataset1 == 'mnist':
			train_set1 = get_dataset(root='./dataset/mnist', dataset='mnist', phase='train')
			test_set1 = get_dataset(root='./dataset/mnist', dataset='mnist', phase='test')
		elif args.dataset1 == 'cifar-10':
			train_set1 = get_dataset(root='./dataset/cifar10', dataset='cifar10', phase='train')
			test_set1 = get_dataset(root='./dataset/cifar10', dataset='cifar10', phase='test')
		elif args.dataset1 == 'cifar-100':
			train_set1 = get_dataset(root='./dataset/cifar100', dataset='cifar100', phase='train')
			test_set1 = get_dataset(root='./dataset/cifar100', dataset='cifar100', phase='test')
		elif args.dataset1 == 'imagenet':
			train_set1 = get_dataset(root='./dataset/imagenet', dataset='imagenet', phase='train')
			test_set1 = get_dataset(root='./dataset/imagenet', dataset='imagenet', phase='test')

		if args.dataset2 == 'mnist':
			train_set2 = get_dataset(root='./dataset/mnist', dataset='mnist', phase='train')
			test_set2 = get_dataset(root='./dataset/mnist', dataset='mnist', phase='test')
		elif args.dataset2 == 'cifar-10':
			train_set2 = get_dataset(root='./dataset/cifar10', dataset='cifar10', phase='train')
			test_set2 = get_dataset(root='./dataset/cifar10', dataset='cifar10', phase='test')
		elif args.dataset2 == 'cifar-100':
			train_set2 = get_dataset(root='./dataset/cifar100', dataset='cifar100', phase='train')
			test_set2 = get_dataset(root='./dataset/cifar100', dataset='cifar100', phase='test')
		elif args.dataset2 == 'imagenet':
			train_set2 = get_dataset(root='./dataset/imagenet', dataset='imagenet', phase='train')
			test_set2 = get_dataset(root='./dataset/imagenet', dataset='imagenet', phase='test')

		self.train_loader1 = DataLoader(train_set1, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
		self.test_loader1 = DataLoader(test_set1, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

		self.train_loader2 = DataLoader(train_set2, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
		self.test_loader2 = DataLoader(test_set2, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

		# Evaluation mode
		if args.eval and args.resume:
			print(f'=> loading checkpoint: {args.resume}')
			checkpoint = torch.load(args.resume)
			epoch = checkpoint['epoch']
			best_loss = checkpoint['best_loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print(f'=> loaded checkpoint: (epoch {epoch})')
			print(f'=> best loss: {best_loss}')

	def train(self):
		start_time = time.time()
		best_loss = np.inf

		for epoch in range(self.epochs):
			loss = self.train_one_epoch(self.train_loader1, self.train_loader2, epoch)
			# loss, acc = self.val_one_epoch(self.val_loader)

			state = {'epoch': epoch + 1,
					 'state_dict': self.model.state_dict(),
					 'best_loss': best_loss,
					 'optimizer': self.optimizer.state_dict()}

			if (epoch+1) % 10 == 0:
				torch.save(state, os.path.join(self.save, 'checkpoint_{}.pth.tar'.format(epoch)))

			if loss < best_loss:
				best_loss = loss
				torch.save(state, os.path.join(self.save, 'best_loss.pth.tar'))

		elapsed_time = time.time() - start_time
		print('---------------------------------------'*2)
		print(f'Training finished! Model: {self.args.model}, Mode: {self.args.mode}')
		print('---------------------------------------'*2)
		print(f'=> Total Epochs: {self.epochs}')
		print(f'=> Best Loss: {best_loss}')
		print(f"=> Total elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
		print('---------------------------------------'*2)

	def train_one_epoch(self, train_loader1, train_loader2, epoch):
		losses = AverageMeter()
		converge = ConvergenceChecker(threshold=1e-2)
		loss = np.inf

		self.model.train()

		for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(train_loader1, train_loader2)):
			data_time.update(time.time() - cur)

			inputs1 = inputs1.to(self.device)
			targets1 = targets1.to(self.device)
			inputs2 = inputs2.to(self.device)
			targets2 = targets2.to(self.device)

			self.optimizer.zero_grad()
			outputs1 = self.fc1(self.model(inputs1))
			outputs2 = self.fc2(self.model(inputs2))

			if self.costout and converge.check(loss):
				rand_num = np.random.rand()

				if rand_num > 0.5:
					loss = self.criterion(outputs1, targets1)
				else:
					loss = self.criterion(outputs2, targets2)
			else:
				loss1 = self.criterion(outputs1, targets1)
				loss2 = self.criterion(outputs2, targets2)
				loss = loss1 + loss2

			loss.backward()
			self.optimizer.step()
			losses.update(loss.item(), inputs.size(0))

			if self.verbose and idx % 100 == 0:
				print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]',
					  f'Loss {losses.val:.4f} ({losses.avg:.4f})',
				)

		return losses.avg

	def val_one_epoch(self, val_loader1, val_loader2):
		losses = AverageMeter()

		self.model.eval()

		with torch.no_grad():
			for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(val_loader1, val_loader2)):
				inputs1 = inputs1.to(self.device)
				inputs2 = inputs2.to(self.device)
				targets1 = targets1.to(self.device)
				targets2 = targets2.to(self.device)
			
				outputs1 = self.fc1(self.model(inputs1))
				outputs2 = self.fc2(self.model(inputs2))

				loss1 = self.criterion(outputs1, targets1)
				loss2 = self.criterion(outputs2, targets2)
				loss = loss1 + loss2
				
				losses.update(loss.item(), inputs.size(0))

				if self.verbose and idx % 100 == 0:
					print(f'Val: [{idx}/{len(val_loader)}]',
					      f'Loss {losses.val:.4f} ({losses.avg:.4f})',
					)

		return losses.avg

	def test(self, test_loader1, test_loader2):
		losses = AverageMeter()

		self.model.eval()

		with torch.no_grad():
			for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(test_loader1, test_loader2)):
				inputs1 = inputs1.to(self.device)
				inputs2 = inputs2.to(self.device)
				targets1 = targets1.to(self.device)
				targets2 = targets2.to(self.device)
		
				outputs1 = self.fc1(self.model(inputs1))
				outputs2 = self.fc2(self.model(inputs2))

				loss1 = self.criterion(outputs1, targets1)
				loss2 = self.criterion(outputs2, targets2)
				loss = loss1 + loss2

				losses.update(loss.item(), inputs.size(0))

				if self.verbose and idx % 10 == 0:
				    print(f'Test: [{idx}/{len(test_loader)}]',
				          f'Loss {losses.val:.4f} ({losses.avg:.4f})',
				    )