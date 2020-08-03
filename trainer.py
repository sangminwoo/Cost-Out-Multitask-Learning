import os
import random
import json
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Model
from models.mlp_multitask import MLP, build_mlp
from models.resnet_multitask import ResNet, build_resnet
# Dataset
from dataset import get_dataset
# Train
from utility.train_util import AverageMeter, ConvergenceChecker, accuracy

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
		self.threshold = args.threshold
		self.start = 0
		self.end = epoch
		self.best_loss = np.inf
		self.best_acc = 0

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
			logger.info(f"=> loading checkpoint: {args.checkpoint}")
			checkpoint = torch.load(args.checkpoint)
			self.start = checkpoint['epoch']
			self.end = self.end + self.start
			self.best_loss = checkpoint['best_loss']
			self.best_acc = checkpoint['best_acc']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			logger.info(f"=> loaded checkpoint: (epoch {self.start})")
			logger.info(f"=> best loss: {self.best_loss:.4f}")
			logger.info(f"=> best acc: {self.best_acc:.4f}")

	def train(self):
		logger = logging.getLogger("cost_out.trainer")
		logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training start!")

		self.converge = ConvergenceChecker(threshold=self.threshold)
		start_time = time.time()

		for epoch in range(self.start, self.end):
			loss, acc = self.train_one_epoch(epoch)
			# loss, acc = self.val_one_epoch(self.val_loader)

			state = {'epoch': epoch + 1,
					 'state_dict': self.model.state_dict(),
					 'best_loss': self.best_loss,
					 'best_acc': self.best_acc,
					 'optimizer': self.optimizer.state_dict()}

			if (epoch+1) % 10 == 0:
				torch.save(state, os.path.join(self.save, 'checkpoint_{}.pth.tar'.format(epoch)))

			if loss < self.best_loss:
				self.best_loss = loss
				self.best_acc = acc
				torch.save(state, os.path.join(self.save, 'best_loss.pth.tar'))

		elapsed_time = time.time() - start_time
		logger.info('---------------------------------------'*2)
		logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training finished!")
		logger.info(f"Dataset1: {self.args.dataset1}, Dataset2: {self.args.dataset2}")
		logger.info(f"Model: {self.args.model}, Costout: {self.costout}")
		logger.info("---------------------------------------"*2)
		logger.info(f"=> Total Epoch: {self.end}")
		logger.info(f"=> Best Loss: {self.best_loss:.4f}")
		logger.info(f"=> Best Acc: {self.best_acc:.4f}")
		logger.info(f"=> Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
		logger.info('---------------------------------------'*2)

	def train_one_epoch(self, epoch):
		logger = logging.getLogger("cost_out.trainer")
		logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		clock = AverageMeter()
		losses = AverageMeter()
		acc = AverageMeter()
		max_iter = len(self.dataloader1)

		self.model.train()

		end = time.time()
		for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
			inputs1 = inputs1.to(self.device)
			targets1 = targets1.to(self.device)
			inputs2 = inputs2.to(self.device)
			targets2 = targets2.to(self.device)

			self.optimizer.zero_grad()
			outputs1 = self.model(inputs1)
			outputs2 = self.model(inputs2)

			if self.costout and self.converge.check(losses.avg):
				# print("loss converged... using costout...")
				rand = np.random.rand()

				if rand > 0.5:
					loss = self.criterion(outputs1, targets1)
				else:
					loss = self.criterion(outputs2, targets2)
			else:
				# print('baseline')
				loss1 = self.criterion(outputs1, targets1)
				loss2 = self.criterion(outputs2, targets2)
				loss = loss1 + loss2

			loss.backward()
			self.optimizer.step()
			losses.update(loss.item(), inputs1.size(0))

			acc1 = accuracy(outputs1, targets1)
			acc2 = accuracy(outputs2, targets2)
			acc_all = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]
			acc.update(acc_all[0].item(), inputs1.size(0))

			batch_time = time.time() - end
			end = time.time()
			clock.update(batch_time)

			eta_seconds = clock.avg * (max_iter - idx) + clock.avg * max_iter * (self.end - epoch)
			eta_string = str(timedelta(seconds=int(eta_seconds)))


			delimeter = "   "
			if self.verbose and idx % 100 == 0:
				logger.info(
					delimeter.join(
						["eta: {eta}",
						 "iter [{epoch}][{idx}/{iter}]",
						 "loss {loss_val:.4f} ({loss_avg:.4f})",
						 "accuracy {acc_val:.4f} ({acc_avg:.4f})"]
					 ).format(
					 		eta=eta_string,
						    epoch=epoch,
						    idx=idx,
						    iter=max_iter,
						    loss_val=losses.val,
						    loss_avg=losses.avg,
						    acc_val=acc.val,
						    acc_avg=acc.avg)
				)

		return losses.avg, acc.avg

	def val_one_epoch(self):
		logger = logging.getLogger("cost_out.inference")
		logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Validation")

		losses = AverageMeter()
		acc = AverageMeter()

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
				acc1 = accuracy(outputs1, targets1)
				acc2 = accuracy(outputs2, targets2)
				acc_all = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]
				acc.update(acc_all[0].item(), inputs1.size(0))

				if self.verbose and idx % 100 == 0:
					logger.info(f"Val: [{idx}/{len(self.dataloader1)}]",
					      		f"Loss {losses.val:.4f} ({losses.avg:.4f})",
					      		f"Accuracy {acc.val:.4f} ({acc.avg:.4f})",
					)

		return losses.avg

	def test(self):
		logger = logging.getLogger("cost_out.inference")
		logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Evaluation start!")

		losses = AverageMeter()
		acc = AverageMeter()

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
				acc1 = accuracy(outputs1, targets1)
				acc2 = accuracy(outputs2, targets2)
				acc_all = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]
				acc.update(acc_all[0].item(), inputs1.size(0))

				if self.verbose and idx % 10 == 0:
				    logger.info(f"Test: [{idx}/{len(self.dataloader1)}]",
				          		f"Loss {losses.val:.4f} ({losses.avg:.4f})",
				          		f"Accuracy {acc.val:.4f} ({acc.avg:.4f})",
				    )