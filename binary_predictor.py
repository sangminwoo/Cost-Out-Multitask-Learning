import argparse
import random
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # DEBUG, INFO, ERROR, WARNING
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def get_timestamp():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
    return st

def get_arguments():
	parser = argparse.ArgumentParser(description='Binary Predictor')
	parser.add_argument('--mode', type=str, help='st; mt', required=True)
	parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
	parser.add_argument('--epoch', type=int, help='the number of epochs', default=1000)
	parser.add_argument('--batch', type=int, help='batch size', default=128)
	parser.add_argument('--len', type=int, help='sequence length', default=10)
	parser.add_argument('--layers', type=int, help='the number of layers', default=2)
	parser.add_argument('--in_dim', type=int, help='size of input dimension', default=64)
	parser.add_argument('--hid_dim', type=int, help='size of hidden dimension', default=1024)
	parser.add_argument('--seed', type=int, help='random seed', default=0)
	args = parser.parse_args()
	return args

def random_number_generator(seq_len=10):
	dim = 2**seq_len
	rand_num = np.arange(dim)
	np.random.shuffle(rand_num)
	rand_num = np.array([bin(rand)[2:] for rand in rand_num])
	
	bin_num = torch.zeros(len(rand_num), seq_len)
	for i in range(len(rand_num)):
		for j in range(len(rand_num[i])):
			bin_num[i, -j-1] = int(rand_num[i][-j-1])
	return bin_num

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BinaryPredictor(nn.Module):
	def __init__(self, mode='mt', seq_len=10, num_layers=10, input_dim=16, hidden_dim=1024, dropout=.0, sigmoid=True):
		super(BinaryPredictor, self).__init__()
		self.mode = mode
		self.seq_len = seq_len
		self.num_seq = 2**seq_len
		self.num_layers = num_layers
		self.sigmoid = sigmoid

		self.input_embed = nn.Embedding(num_embeddings=2**seq_len, embedding_dim=input_dim)

		self.st_input = nn.ModuleList([
							nn.Sequential(
								nn.Linear(input_dim, hidden_dim),
		                        nn.BatchNorm1d(hidden_dim),
		                        nn.ReLU(True),
		                        nn.Dropout(p=dropout)
		                   	) for _ in range(self.seq_len)
						])
		self.mt_input = nn.Sequential(
							nn.Linear(input_dim, hidden_dim),
		                    nn.BatchNorm1d(hidden_dim),
		                    nn.ReLU(True),
		                    nn.Dropout(p=dropout)
		               	)

		if self.num_layers >= 2:
			self.st_hidden = nn.ModuleList([
							  nn.ModuleList([
								  nn.Sequential(
								  	  nn.Linear(hidden_dim, hidden_dim),
									  nn.BatchNorm1d(hidden_dim),
									  nn.ReLU(True),
									  nn.Dropout(p=dropout)
								  ) for _ in range(num_layers-2)
							  ]) for _ in range(self.seq_len)
						  ])
			self.mt_hidden = nn.ModuleList([
							  nn.Sequential(
							  	  nn.Linear(hidden_dim, hidden_dim),
								  nn.BatchNorm1d(hidden_dim),
								  nn.ReLU(True),
								  nn.Dropout(p=dropout)
							  ) for _ in range(num_layers-2)
						  ])

		self.st_output = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.seq_len)])
		self.mt_output = nn.Linear(hidden_dim, self.seq_len)

		# for m in self.modules():
		# 	if isinstance(m, nn.Linear):
		# 		nn.init.kaiming_uniform_(m.weight)
		# 		nn.init.kaiming_uniform_(m.bias)

	def forward(self, x):
		x = self.input_embed(x)
		if self.mode == 'st':
			out = []
			for i in range(self.seq_len):
				y = self.st_input[i](x)	# input layer
				for hidden in self.st_hidden[i]:
					y = hidden(y)		# hidden layer
				y = self.st_output[i](y)	# output layer
				out.append(y)
			out = torch.cat(out, dim=1)

		elif self.mode == 'mt':
			y = self.mt_input(x)
			for hidden in self.mt_hidden:
				y = hidden(y)
			out = self.mt_output(y)

		return torch.sigmoid(out) if self.sigmoid else out

if __name__ == '__main__':
	args = get_arguments()
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	cudnn.deterministic = True

	num_seq=2**args.len
	logger = setup_logger(name=args.mode, save_dir='logs', filename='{}_binary_predictor_{}.txt'.format(get_timestamp(), args.mode))
	logger = logging.getLogger(args.mode)
		
	model = BinaryPredictor(mode=args.mode, seq_len=args.len, num_layers=args.layers, input_dim=args.in_dim,
							hidden_dim=args.hid_dim, dropout=0., sigmoid=True)
	input_nums = torch.arange(num_seq)#.float().reshape(-1, 1)
	target_nums = random_number_generator(seq_len=args.len)

	dataloader = DataLoader(TensorDataset(input_nums, target_nums), batch_size=args.batch, shuffle=True, num_workers=8)

	criterion = nn.L1Loss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
	
	model = nn.DataParallel(model, device_ids=[0,1,2,3])
	model = model.cuda()
	model.train()

	losses = AverageMeter()
	best_loss = np.inf

	logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training start!")
	start_time = time.time()

	for epoch in range(args.epoch):
		for idx, (input, target) in enumerate(dataloader):
			input = input.cuda()
			target = target.cuda()
			output = model(input)

			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			losses.update(loss.item(), input.size(0))

			if losses.val < best_loss:
				best_loss_epoch = epoch+1
				best_loss = losses.val

			delimeter = "   "
			logger.info(
				delimeter.join(
					["iter [{epoch}][{idx}/{iter}]",
					 # "input {input}",
					 # "output {output}",
					 # "target {target}",
					 "loss {loss_val:.4f} ({loss_avg:.4f})"]
				 ).format(
						epoch=epoch+1,
						idx=idx,
						iter=len(dataloader),
						# input=input,
						# output=output,
						# target=target,	
					    loss_val=losses.val,
					    loss_avg=losses.avg)
			)

	elapsed = time.time() - start_time
	logger.info('---------------------------------------'*2)
	logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training finished!")
	logger.info(f"elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
	logger.info(f"epoch:{args.epoch}")
	logger.info(f"best_loss:{best_loss:.4f} in {best_loss_epoch}th epoch")
	logger.info('---------------------------------------'*2)