import os
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Model
from models.mlp_multitask import MLP, build_mlp
from models.resnet_multitask import ResNet, build_resnet
# Dataset
from dataset import get_dataset
from torch.utils.data import DataLoader
# Validation
from utils import AverageMeter, ConvergenceChecker, accuracy

class Trainer:
	def __init__(self, args, epoch=100, model='resnet18', optimizer='SGD',	verbose=True):
	
		assert model in ['mlp', 'resnet18', 'resnet50', 'resnet101'], 'model not available!'
		assert optimizer in ['SGD', 'Adam'], 'optimizer not available!'

		self.args = args
		self.dropout = args.dropout
		self.device = args.device
		self.save = args.save
		self.verbose = args.verbose
		self.costout = args.costout
		self.start = 0
		self.end = epoch
		self.best_loss = np.inf

		# Dataset / DataLoader
		if args.dataset1 == 'mnist':
			dataset1 = get_dataset(root='./data/mnist', dataset='mnist', phase='train' if not args.eval else 'test')
			num_classes1 = 10
		elif args.dataset1 == 'cifar-10':
			dataset1 = get_dataset(root='./data/cifar10', dataset='cifar10', phase='train' if not args.eval else 'test')
			num_classes1 = 10
		elif args.dataset1 == 'cifar-100':
			dataset1 = get_dataset(root='./data/cifar100', dataset='cifar100', phase='train' if not args.eval else 'test')
			num_classes1 = 100
		elif args.dataset1 == 'imagenet':
			dataset1 = get_dataset(root='./data/imagenet', dataset='imagenet', phase='train' if not args.eval else 'test')
			num_classes1 = 1000

		if args.dataset2 == 'mnist':
			dataset2 = get_dataset(root='./data/mnist', dataset='mnist', phase='train' if not args.eval else 'test')
			num_classes2 = 10
		elif args.dataset2 == 'cifar-10':
			dataset2 = get_dataset(root='./data/cifar10', dataset='cifar10', phase='train' if not args.eval else 'test')
			num_classes2 = 10
		elif args.dataset2 == 'cifar-100':
			dataset2 = get_dataset(root='./data/cifar100', dataset='cifar100', phase='train' if not args.eval else 'test')
			num_classes2 = 100
		elif args.dataset2 == 'imagenet':
			dataset2 = get_dataset(root='./data/imagenet', dataset='imagenet', phase='train' if not args.eval else 'test')
			num_classes2 = 1000

		print(dataset1); print(dataset2)
		self.dataloader1 = DataLoader(dataset1, args.batch, shuffle=True if not args.eval else False, num_workers=args.workers, pin_memory=True)
		self.dataloader2 = DataLoader(dataset2, args.batch, shuffle=True if not args.eval else False, num_workers=args.workers, pin_memory=True)

		# Model
		if model == 'mlp':
			self.model = build_mlp(10, 512, 128, args.bn_momentum, args.dropout)
		elif model == 'resnet18':
			self.model = build_resnet(arch='resnet18', pretrained=False)
		elif model == 'resnet50':
			self.model = build_resnet(arch='resnet50', pretrained=False)
		elif model == 'resnet101':
			self.model = build_resnet(arch='resnet101', pretrained=False)
		self.fc1 = nn.Linear(512 * self.model.block.expansion, num_classes1)
		self.fc2 = nn.Linear(512 * self.model.block.expansion, num_classes2)

		self.model = nn.DataParallel(self.model.to(self.device))
		self.fc1 = nn.DataParallel(self.fc1.to(self.device))
		self.fc2 = nn.DataParallel(self.fc2.to(self.device))

		self.criterion = nn.CrossEntropyLoss().to(self.device)
		if optimizer == 'SGD':
			self.optimizer = optim.SGD(	self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		# Resume
		if args.eval or args.resume:
			print(f'=> loading checkpoint: {args.checkpoint}')
			checkpoint = torch.load(args.checkpoint)
			self.start = checkpoint['epoch']
			self.best_loss = checkpoint['best_loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print(f'=> loaded checkpoint: (epoch {self.start})')
			print(f'=> best loss: {self.best_loss}')

	def train(self):
		start_time = time.time()

		for epoch in range(self.start, self.start+self.end):
			loss = self.train_one_epoch(epoch)
			# loss, acc = self.val_one_epoch(self.val_loader)

			state = {'epoch': epoch + 1,
					 'state_dict': self.model.state_dict(),
					 'best_loss': self.best_loss,
					 'optimizer': self.optimizer.state_dict()}

			if (epoch+1) % 10 == 0:
				torch.save(state, os.path.join(self.save, 'checkpoint_{}.pth.tar'.format(epoch)))

			if loss < self.best_loss:
				self.best_loss = loss
				torch.save(state, os.path.join(self.save, 'best_loss.pth.tar'))

		elapsed_time = time.time() - start_time
		mode = 'Costout' if self.costout else 'Baseline' 
		print('---------------------------------------'*2)
		print(f'Training finished! Model: {self.args.model}, Mode: {mode}')
		print('---------------------------------------'*2)
		print(f'=> Total Epoch: {self.start+self.end}')
		print(f'=> Best Loss: {self.best_loss}')
		print(f"=> Total elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
		print('---------------------------------------'*2)

	def train_one_epoch(self, epoch):
		losses = AverageMeter()
		converge = ConvergenceChecker(threshold=1e-2)
		loss = np.inf

		self.model.train()

		for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
			inputs1 = inputs1.to(self.device)
			targets1 = targets1.to(self.device)
			inputs2 = inputs2.to(self.device)
			targets2 = targets2.to(self.device)

			self.optimizer.zero_grad()
			outputs1 = self.model(inputs1)
			outputs2 = self.model(inputs2)

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
			losses.update(loss.item(), inputs1.size(0))

			if self.verbose and idx % 100 == 0:
				print(f'Epoch: [{epoch}][{idx}/{len(self.dataloader1)}]',
					  f'Loss {losses.val:.4f} ({losses.avg:.4f})',
				)

		return losses.avg

	def val_one_epoch(self):
		losses = AverageMeter()

		self.model.eval()

		with torch.no_grad():
			for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
				inputs1 = inputs1.to(self.device)
				inputs2 = inputs2.to(self.device)
				targets1 = targets1.to(self.device)
				targets2 = targets2.to(self.device)
			
				outputs1 = self.model(inputs1)
				outputs2 = self.model(inputs2)

				loss1 = self.criterion(outputs1, targets1)
				loss2 = self.criterion(outputs2, targets2)
				loss = loss1 + loss2
				
				losses.update(loss.item(), inputs1.size(0))

				if self.verbose and idx % 100 == 0:
					print(f'Val: [{idx}/{len(self.dataloader1)}]',
					      f'Loss {losses.val:.4f} ({losses.avg:.4f})',
					)

		return losses.avg

	def test(self):
		losses = AverageMeter()

		self.model.eval()

		with torch.no_grad():
			for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
				inputs1 = inputs1.to(self.device)
				inputs2 = inputs2.to(self.device)
				targets1 = targets1.to(self.device)
				targets2 = targets2.to(self.device)
		
				outputs1 = self.model(inputs1)
				outputs2 = self.model(inputs2)

				loss1 = self.criterion(outputs1, targets1)
				loss2 = self.criterion(outputs2, targets2)
				loss = loss1 + loss2

				losses.update(loss.item(), inputs1.size(0))

				if self.verbose and idx % 10 == 0:
				    print(f'Test: [{idx}/{len(self.dataloader1)}]',
				          f'Loss {losses.val:.4f} ({losses.avg:.4f})',
				    )