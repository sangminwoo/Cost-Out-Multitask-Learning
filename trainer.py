import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from mlp import build_mlp
from utils import AverageMeter, accuracy
from dataset import get_dataset
loader import ImageLoader, get_dataloader
import json

class Trainer:
	def __init__(self, args, epochs=100, model='resnet18', optimizer='SGD',	verbose=True):
	
		assert model in ['mlp', 'resnet18', 'resnet50', 'resnet101'], 'Can not use that model!'
		assert optimizer in ['SGD', 'Adam'], 'Can not use that optimizer!'

		self.args = args
		self.epochs = epochs
		self.dropout = args.dropout
		self.device = args.device
		self.save = args.save
		self.verbose = args.verbose
		self.costout = args.costout

		# Model
		if model == 'mlp':
			self.model = build_mlp(10, 1568, 2048, 10, args.bn_momentum, args.dropout)
		elif model == 'resnet18':
			self.model = torchvision.models.resnet18(pretrained=False, progress=True)
		elif model == 'resnet50':
			self.model = torchvision.models.resnet18(pretrained=False, progress=True)
		elif model == 'resnet101':
			self.model = torchvision.models.resnet101(pretrained=False,	progress=True)

		self.model = nn.DataParallel(self.model.to(self.device))
		self.criterion = nn.CrossEntropyLoss().to(self.device)
		if optimizer == 'SGD':
			self.optimizer = optim.SGD(	self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		# Dataset / DataLoader
		if args.dataset == 'mnist':
			train_set = get_dataset(root='./dataset/mnist', dataset='mnist', phase='train')
			test_set = get_dataset(root='./dataset/mnist', dataset='mnist', phase='test')
		elif args.dataset == 'cifar-10':
			train_set = get_dataset(root='./dataset/cifar10', dataset='cifar10', phase='train')
			test_set = get_dataset(root='./dataset/cifar10', dataset='cifar10', phase='test')
		elif args.dataset == 'cifar-100':
			train_set = get_dataset(root='./dataset/cifar100', dataset='cifar100', phase='train')
			test_set = get_dataset(root='./dataset/cifar100', dataset='cifar100', phase='test')
		elif args.dataset == 'imagenet':
			train_set = get_dataset(root='./dataset/imagenet', dataset='imagenet', phase='train')
			test_set = get_dataset(root='./dataset/imagenet', dataset='imagenet', phase='test')

		self.train_loader = DataLoader(train_set, batch, shuffle=True, num_workers=workers, pin_memory=True)
		self.test_loader = DataLoader(test_set, batch, shuffle=False, num_workers=workers, pin_memory=True)

		# Evaluation mode
		if args.eval and args.resume:
			print(f'=> loading checkpoint: {args.resume}')
			checkpoint = torch.load(args.resume)
			epoch = checkpoint['epoch']
			best_acc = checkpoint['best_acc']
			best_loss = checkpoint['best_loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print(f'=> loaded checkpoint: (epoch {epoch})')
			print(f'=> best acc: {best_acc}')
			print(f'=> best loss: {best_loss}')

	def train(self):
		start_time = time.time()

		best_loss = np.inf
		best_acc = 0

		for epoch in range(self.epochs):
			loss, acc = self.train_one_epoch(self.train_loader, epoch)
			# loss, acc = self.val_one_epoch(self.val_loader)

			state = {'epoch': epoch + 1,
					 'state_dict': self.model.state_dict(),
					 'best_acc': best_acc
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
		print(f'=> Best Acc: {best_acc}')
		print(f'=> Best Loss: {best_loss}')
		print(f"=> Total elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
		print('---------------------------------------'*2)

	def train_one_epoch(self, train_loader, epoch):
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		acc = AverageMeter()

		self.model.train()
		cur = time.time()

		for idx, (inputs, targets1, targets2) in enumerate(train_loader):
			data_time.update(time.time() - cur)

			inputs = inputs.to(self.device) # Nx1568
			targets1 = targets1.to(self.device)
			targets2 = targets2.to(self.device)

			self.optimizer.zero_grad()
			output = self.model(inputs)

			if self.costout:
				random_num = np.random.rand()
				if random_num > 0.5:
					loss = self.criterion(output[:, :10], targets1)
					acc1 = accuracy(output[:, :10], targets1)
					acc = acc1
				else:
					loss = self.criterion(output[:, 10:], targets2)
					acc2 = accuracy(output[:, 10:], targets2)
					acc = acc2
			else:
				loss1 = self.criterion(output[:, :10], targets1)
				loss2 = self.criterion(output[:, 10:], targets2)
				loss = (loss1 + loss2) / 2
				acc1 = accuracy(output[:, :10], targets1)
				acc2 = accuracy(output[:, 10:], targets2)
				acc = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]

			loss.backward()
			self.optimizer.step()
			losses.update(loss.item(), inputs.size(0))
			acc.update(acc[0].item(), inputs.size(0))
			
			batch_time.update(time.time() - cur)
			cur = time.time()

			if self.verbose and idx % 100 == 0:
				print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]',
					  f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})',
					  f'Data {data_time.val:.3f} ({data_time.avg:.3f})',
					  f'Loss {losses.val:.4f} ({losses.avg:.4f})',
					  f'Acc {acc.val:.3f} ({acc.avg:.3f})'
				)

		return losses.avg, acc

	def val_one_epoch(self, val_loader):
		batch_time = AverageMeter()
		losses = AverageMeter()
		acc = AverageMeter()

		self.model.eval()
		cur = time.time()

		with torch.no_grad():
			for idx, (inputs, targets1, targets2) in enumerate(val_loader):
				inputs = inputs.to(self.device)
				targets1 = targets1.to(self.device)
				targets2 = targets2.to(self.device)
			
				output = self.model(inputs)

				if self.costout:
					random_num = np.random.rand()
					if random_num > 0.5:
						loss = self.criterion(output[:, :10], targets1)
						acc1 = accuracy(output[:, :10], targets1)
						acc = acc1
					else:
						loss = self.criterion(output[:, 10:], targets2)
						acc2 = accuracy(output[:, 10:], targets2)
						acc = acc2
				else:
					loss1 = self.criterion(output[:, :10], targets1)
					loss2 = self.criterion(output[:, 10:], targets2)
					loss = (loss1 + loss2) / 2
					acc1 = accuracy(output[:, :10], targets1)
					acc2 = accuracy(output[:, 10:], targets2)
					acc = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]
				
				losses.update(loss.item(), inputs.size(0))
				acc.update(acc[0].item(), inputs.size(0))

				batch_time.update(time.time() - cur)
				cur = time.time()

				if self.verbose and idx % 100 == 0:
					print(f'Val: [{idx}/{len(val_loader)}]',
					      f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})',
					      f'Loss {losses.val:.4f} ({losses.avg:.4f})',
					      f'Acc {acc.val:.3f} ({acc.avg:.3f})'
					      )

			print(f'Acc {acc.val:.3f} ({acc.avg:.3f})')

		return losses.avg, acc

	def test(self):
		test_data = MNISTDataset(data_dir=self.data, phase='test')
		test_loader = get_dataloader(test_data, self.batch, self.workers, phase='test')

		batch_time = AverageMeter()
		losses = AverageMeter()
		acc = AverageMeter()

		self.model.eval()
		cur = time.time()

		with torch.no_grad():
			for idx, (inputs, targets1, targets2) in enumerate(test_loader):
				inputs = inputs.to(self.device)
				targets1 = targets1.to(self.device)
				targets2 = targets2.to(self.device)
		
				output = self.model(inputs)

				if self.costout:
					random_num = np.random.rand()
					if random_num > 0.5:
						loss = self.criterion(output[:, :10], targets1)
						acc1 = accuracy(output[:, :10], targets1)
						acc = acc1
					else:
						loss = self.criterion(output[:, 10:], targets2)
						acc2 = accuracy(output[:, 10:], targets2)
						acc = acc2
				else:
					loss1 = self.criterion(output[:, :10], targets1)
					loss2 = self.criterion(output[:, 10:], targets2)
					loss = (loss1 + loss2) / 2
					acc1 = accuracy(output[:, :10], targets1)
					acc2 = accuracy(output[:, 10:], targets2)
					acc = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]

				losses.update(loss.item(), inputs.size(0))
				acc.update(acc[0].item(), inputs.size(0))

				batch_time.update(time.time() - cur)
				cur = time.time()

				if self.verbose and idx % 10 == 0:
				    print(f'Test: [{idx}/{len(test_loader)}]',
				          f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})',
				          f'Loss {losses.val:.4f} ({losses.avg:.4f})',
				          f'Acc {acc.val:.3f} ({acc.avg:.3f})'
				          )

			print(f'Acc {acc.val:.3f} ({acc.avg:.3f})')