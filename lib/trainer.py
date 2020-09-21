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
from lib.models.single_task.mlp import build_mlp
from lib.models.single_task.resnet import build_resnet
from lib.models.multi_task.mlp_multitask import build_mt_mlp
from lib.models.multi_task.resnet_multitask import build_mt_resnet
# Dataset
from lib.dataset import get_dataset
# Train
from lib.utils.train_utils import AverageMeter, ConvergenceChecker, accuracy

class Trainer:
	def __init__(self, cfg, args):
		self.cfg = cfg
		self.args = args
		self.mode = args.mode
		self.device = args.device
		self.save = args.save
		self.threshold = cfg.MODEL.CONVERGENCE_THRESHOLD
		self.verbose = cfg.ETC.VERBOSE
		self.logger = logging.getLogger(args.mode)
		self.start = 0
		self.end = cfg.SOLVER.EPOCH
		self.best_loss = np.inf
		self.best_acc = 0

		# Dataset / DataLoader
		dataset1 = get_dataset(root=os.path.join(cfg.DATASET.PATH, cfg.DATASET.DATA1),
							   dataset=cfg.DATASET.DATA1,
							   phase='train' if not args.eval else 'test')
		self.dataloader1 = DataLoader(dataset1,
									  batch_size=cfg.DATASET.TRAIN_BATCH_SIZE if not args.eval else cfg.DATASET.TEST_BATCH_SIZE,
									  shuffle=True if not args.eval else False,
									  num_workers=cfg.ETC.WORKERS,
									  pin_memory=True)

		if self.mode in ['mt_baseline', 'mt_filter', 'mt_costout']:
			dataset2 = get_dataset(root=os.path.join(cfg.DATASET.PATH, cfg.DATASET.DATA2),
								   dataset=cfg.DATASET.DATA2,
								   phase='train' if not args.eval else 'test')
			self.dataloader2 = DataLoader(dataset2,
										  batch_size=cfg.DATASET.TRAIN_BATCH_SIZE if not args.eval else cfg.DATASET.TEST_BATCH_SIZE,
										  shuffle=True if not args.eval else False,
										  num_workers=cfg.ETC.WORKERS,
										  pin_memory=True)

		# Model
		if self.mode == 'st_baseline':		
			if cfg.MODEL.BASE_MODEL == 'mlp':
				self.model = build_mlp(args, num_layers=5, input_size=1*28*28 if cfg.DATASET.DATA1=='mnist' else 3*224*224, hidden_size=128, output_size=cfg.DATASET.NUM_CLASSES1, bn_momentum=cfg.SOLVER.BN_MOMENTUM, dropout=cfg.SOLVER.DROPOUT)
			elif cfg.MODEL.BASE_MODEL in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
				self.model = build_resnet(args, arch=cfg.MODEL.BASE_MODEL, pretrained=False, num_classes=cfg.DATASET.NUM_CLASSES1)
			else:
				pass
		else:			
			if cfg.MODEL.BASE_MODEL == 'mlp':
				self.model = build_mt_mlp(args, num_layers=5, input_size=1*28*28 if cfg.DATASET.DATA1=='mnist' else 3*224*224, hidden_size=128, output_size1=cfg.DATASET.NUM_CLASSES1, output_size2=cfg.DATASET.NUM_CLASSES2, bn_momentum=cfg.SOLVER.BN_MOMENTUM, dropout=cfg.SOLVER.DROPOUT)
			elif cfg.MODEL.BASE_MODEL in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
				self.model = build_mt_resnet(args, arch=cfg.MODEL.BASE_MODEL, num_classes1=cfg.DATASET.NUM_CLASSES1, num_classes2=cfg.DATASET.NUM_CLASSES2)
			else:
				pass
		self.model = nn.DataParallel(self.model.to(self.device))

		# Criterion & Optimizer
		self.criterion = nn.CrossEntropyLoss().to(self.device)
		if cfg.SOLVER.OPTIMIZER == 'SGD':
			self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
		elif cfg.SOLVER.OPTIMIZER == 'Adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

		# Resume
		if args.eval or args.resume:
			self.logger.info(f"=> loading checkpoint: {args.checkpoint}")
			checkpoint = torch.load(args.checkpoint)
			if not args.eval:
				self.start = checkpoint['epoch']
				self.end = self.end + self.start
			self.best_loss = checkpoint['best_loss']
			self.best_acc = checkpoint['best_acc']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.logger.info(f"=> loaded checkpoint: (epoch {self.start})")
			self.logger.info(f"=> best loss: {self.best_loss:.4f}")
			self.logger.info(f"=> best acc: {self.best_acc:.4f}")

	def train(self):
		self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training start!")
		self.clock = AverageMeter()
		self.converge = ConvergenceChecker(threshold=self.threshold)
		start_time = time.time()

		if self.mode == 'st_baseline':
			for epoch in range(self.start, self.end):
				loss, acc = self.train_one_epoch(epoch)
				# loss, acc = self.val_one_epoch(epoch)

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
			self.logger.info('---------------------------------------'*2)
			self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training finished!")
			self.logger.info(f"Dataset1: {self.cfg.DATASET.DATA1}")
			self.logger.info(f"Model: {self.cfg.MODEL.BASE_MODEL}, Mode: {self.mode}")
			self.logger.info("---------------------------------------"*2)
			self.logger.info(f"=> Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
			self.logger.info(f"=> Total Epoch: {self.end}")
			self.logger.info(f"=> Best Loss: {self.best_loss:.4f}")
			self.logger.info(f"=> Best Acc: {self.best_acc:.4f}")
			self.logger.info('---------------------------------------'*2)
		else:
			for epoch in range(self.start, self.end):
				loss, acc1, acc2 = self.train_one_epoch(epoch)
				# loss, acc1, acc2 = self.val_one_epoch(epoch)

				state = {'epoch': epoch + 1,
						 'state_dict': self.model.state_dict(),
						 'best_loss': self.best_loss,
						 'best_acc1': self.best_acc1,
						 'best_acc2': self.best_acc2,
						 'optimizer': self.optimizer.state_dict()}

				if (epoch+1) % 10 == 0:
					torch.save(state, os.path.join(self.save, 'checkpoint_{}.pth.tar'.format(epoch)))

				if loss < self.best_loss:
					self.best_loss = loss
					self.best_acc1 = acc1
					self.best_acc2 = acc2
					torch.save(state, os.path.join(self.save, 'best_loss.pth.tar'))

			elapsed_time = time.time() - start_time
			self.logger.info('---------------------------------------'*2)
			self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training finished!")
			self.logger.info(f"Dataset1: {self.cfg.DATASET.DATA1}")
			self.logger.info(f"Dataset2: {self.cfg.DATASET.DATA2}")
			self.logger.info(f"Model: {self.cfg.MODEL.BASE_MODEL}, Mode: {self.mode}")
			self.logger.info("---------------------------------------"*2)
			self.logger.info(f"=> Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
			self.logger.info(f"=> Total Epoch: {self.end}")
			self.logger.info(f"=> Best Loss: {self.best_loss:.4f}")
			self.logger.info(f"=> Best Acc1: {self.best_acc1:.4f}")
			self.logger.info(f"=> Best Acc2: {self.best_acc2:.4f}")
			self.logger.info('---------------------------------------'*2)		

	def train_one_epoch(self, epoch):
		losses = AverageMeter()
		max_iter = len(self.dataloader1)
		self.model.train()
		end = time.time()

		if self.mode == 'st_baseline':
			acc = AverageMeter()

			for idx, (inputs1, targets1) in enumerate(self.dataloader1):
				inputs1 = inputs1.to(self.device)
				targets1 = targets1.to(self.device)
				self.optimizer.zero_grad()
				outputs1 = self.model(inputs1)
				loss = self.criterion(outputs1, targets1)
				loss.backward()
				self.optimizer.step()
				losses.update(loss.item(), inputs1.size(0))

				a1 = accuracy(outputs1, targets1)
				acc.update(a1[0].item(), inputs1.size(0))

				batch_time = time.time() - end
				end = time.time()
				self.clock.update(batch_time)

				delimeter = "   "
				if self.verbose and idx % 100 == 0:
					eta_seconds = self.clock.avg * (max_iter-idx) + self.clock.avg * max_iter * (self.end-epoch-1)
					eta_string = str(timedelta(seconds=int(eta_seconds)))

					self.logger.info(
						delimeter.join(
							["eta: {eta}",
							 "iter [{epoch}][{idx}/{iter}]",
							 "loss {loss_val:.4f} ({loss_avg:.4f})",
							 "accuracy {acc_val:.4f} ({acc_avg:.4f})"]
						 ).format(
						 		eta=eta_string,
							    epoch=epoch+1,
							    idx=idx,
							    iter=max_iter,
							    loss_val=losses.val,
							    loss_avg=losses.avg,
							    acc_val=acc.val,
							    acc_avg=acc.avg)
					)

			return losses.avg, acc.avg

		else:
			acc1 = AverageMeter()
			acc2 = AverageMeter()

			for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
				inputs1 = inputs1.to(self.device)
				targets1 = targets1.to(self.device)
				inputs2 = inputs2.to(self.device)
				targets2 = targets2.to(self.device)

				self.optimizer.zero_grad()
				outputs1, outputs2 = self.model(inputs1, inputs2)

				if self.mode == 'costout' and self.converge.check(losses.avg):
					# self.logger.info("loss converged... using costout...")
					rand = np.random.rand()

					if rand > 0.5:
						loss = self.criterion(outputs1, targets1)
					else:
						loss = self.criterion(outputs2, targets2)
				else:
					# self.logger.info('baseline')
					loss1 = self.criterion(outputs1, targets1)
					loss2 = self.criterion(outputs2, targets2)
					loss = loss1 + loss2

				loss.backward()
				self.optimizer.step()
				losses.update(loss.item(), inputs1.size(0))

				a1 = accuracy(outputs1, targets1)
				a2 = accuracy(outputs2, targets2)
				acc1.update(a1[0].item(), inputs1.size(0))
				acc2.update(a2[0].item(), inputs2.size(0))

				batch_time = time.time() - end
				end = time.time()
				self.clock.update(batch_time)

				delimeter = "   "
				if self.verbose and idx % 100 == 0:
					eta_seconds = self.clock.avg * (max_iter-idx) + self.clock.avg * max_iter * (self.end-epoch-1)
					eta_string = str(timedelta(seconds=int(eta_seconds)))

					self.logger.info(
						delimeter.join(
							["eta: {eta}",
							 "iter [{epoch}][{idx}/{iter}]",
							 "loss {loss_val:.4f} ({loss_avg:.4f})",
							 "accuracy1 {acc1_val:.4f} ({acc1_avg:.4f})",
							 "accuracy2 {acc2_val:.4f} ({acc2_avg:.4f})",]
						 ).format(
						 		eta=eta_string,
							    epoch=epoch+1,
							    idx=idx,
							    iter=max_iter,
							    loss_val=losses.val,
							    loss_avg=losses.avg,
							    acc1_val=acc1.val,
							    acc1_avg=acc1.avg,
							    acc2_val=acc2.val,
							    acc2_avg=acc2.avg)
					)

			return losses.avg, acc1.avg, acc2.avg

	def val_one_epoch(self, epoch):
		self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Validation")

		losses = AverageMeter()
		acc = AverageMeter()
		max_iter = len(self.dataloader1)

		self.model.eval()

		with torch.no_grad():
			if self.mode == 'st_baseline':
				for idx, (inputs1, targets1) in enumerate(self.dataloader1):
					inputs1 = inputs1.to(self.device)
					targets1 = targets1.to(self.device)
					outputs1 = self.model(inputs1)
					loss = self.criterion(outputs1, targets1)
					losses.update(loss.item(), inputs1.size(0))
					acc1 = accuracy(outputs1, targets1)
					acc.update(acc1[0].item(), inputs1.size(0))

					delimeter = "   "
					if self.verbose and idx % 100 == 0:
						self.logger.info(
							delimeter.join(
								["iter [{epoch}][{idx}/{iter}]",
								 "loss {loss_val:.4f} ({loss_avg:.4f})",
								 "accuracy {acc_val:.4f} ({acc_avg:.4f})"]
							 ).format(
								    epoch=epoch,
								    idx=idx,
								    iter=max_iter,
								    loss_val=losses.val,
								    loss_avg=losses.avg,
								    acc_val=acc.val,
								    acc_avg=acc.avg)
						)
			else:
				for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
					inputs1 = inputs1.to(self.device)
					inputs2 = inputs2.to(self.device)
					targets1 = targets1.to(self.device)
					targets2 = targets2.to(self.device)
				
					outputs1, outputs2 = self.model(inputs1, outputs2)

					loss1 = self.criterion(outputs1, targets1)
					loss2 = self.criterion(outputs2, targets2)
					loss = loss1 + loss2
					
					losses.update(loss.item(), inputs1.size(0))
					acc1 = accuracy(outputs1, targets1)
					acc2 = accuracy(outputs2, targets2)
					acc_all = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]
					acc.update(acc_all[0].item(), inputs1.size(0))

					delimeter = "   "
					if self.verbose and idx % 100 == 0:
						self.logger.info(
							delimeter.join(
								["iter [{epoch}][{idx}/{iter}]",
								 "loss {loss_val:.4f} ({loss_avg:.4f})",
								 "accuracy {acc_val:.4f} ({acc_avg:.4f})"]
							 ).format(
								    epoch=epoch,
								    idx=idx,
								    iter=max_iter,
								    loss_val=losses.val,
								    loss_avg=losses.avg,
								    acc_val=acc.val,
								    acc_avg=acc.avg)
						)

		return losses.avg

	def test(self):
		self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Evaluation start!")

		losses = AverageMeter()
		acc = AverageMeter()
		max_iter = len(self.dataloader1)

		self.model.eval()

		with torch.no_grad():
			if self.mode == 'st_baseline':
				for idx, (inputs1, targets1) in enumerate(self.dataloader1):
					inputs1 = inputs1.to(self.device)
					targets1 = targets1.to(self.device)			
					outputs1 = self.model(inputs1)
					loss1 = self.criterion(outputs1, targets1)
					losses.update(loss1.item(), inputs1.size(0))
					acc1 = accuracy(outputs1, targets1)
					acc.update(acc1[0].item(), inputs1.size(0))

					delimeter = "   "
					if self.verbose and idx % 100 == 0:
						self.logger.info(
							delimeter.join(
								["iter [{idx}/{iter}]",
								 "loss {loss_val:.4f} ({loss_avg:.4f})",
								 "accuracy {acc_val:.4f} ({acc_avg:.4f})"]
							 ).format(
								    idx=idx,
								    iter=max_iter,
								    loss_val=losses.val,
								    loss_avg=losses.avg,
								    acc_val=acc.val,
								    acc_avg=acc.avg)
						)
			else:
				for idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(self.dataloader1, self.dataloader2)):
					inputs1 = inputs1.to(self.device)
					inputs2 = inputs2.to(self.device)
					targets1 = targets1.to(self.device)
					targets2 = targets2.to(self.device)
			
					outputs1, outputs2 = self.model(inputs1, inputs2)

					loss1 = self.criterion(outputs1, targets1)
					loss2 = self.criterion(outputs2, targets2)
					loss = loss1 + loss2

					losses.update(loss.item(), inputs1.size(0))
					acc1 = accuracy(outputs1, targets1)
					acc2 = accuracy(outputs2, targets2)
					acc_all = [(a1 + a2) / 2 for a1, a2 in zip(acc1, acc2)]
					acc.update(acc_all[0].item(), inputs1.size(0))

					delimeter = "   "
					if self.verbose and idx % 100 == 0:
						self.logger.info(
							delimeter.join(
								["iter [{idx}/{iter}]",
								 "loss {loss_val:.4f} ({loss_avg:.4f})",
								 "accuracy {acc_val:.4f} ({acc_avg:.4f})"]
							 ).format(
								    idx=idx,
								    iter=max_iter,
								    loss_val=losses.val,
								    loss_avg=losses.avg,
								    acc_val=acc.val,
								    acc_avg=acc.avg)
						)

		self.logger.info('---------------------------------------'*2)
		self.logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Evaluation finished!")
		self.logger.info(f"Dataset1: {self.cfg.DATASET.DATA1}, Dataset2: {self.cfg.DATASET.DATA2}")
		self.logger.info(f"Model: {self.cfg.MODEL.BASE_MODEL}, Mode: {self.mode}")
		self.logger.info("---------------------------------------"*2)
		self.logger.info(f"=> Loss: {losses.avg:.4f}")
		self.logger.info(f"=> Acc: {acc.avg:.4f}")
		self.logger.info('---------------------------------------'*2)