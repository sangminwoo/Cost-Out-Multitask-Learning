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

class STBinaryPredictor(nn.Module):
	def __init__(self,
				 seq_len=10,
				 num_layers=10,
				 input_dim=16,
				 hidden_dim=1024,
				 dropout=.0,
				 sigmoid=True):

		super(STBinaryPredictor, self).__init__()
		self.seq_len = seq_len
		self.sigmoid = sigmoid

		self.input_embed = nn.Embedding(num_embeddings=2**self.seq_len, embedding_dim=input_dim)

		self.st_input = nn.ModuleList([
							nn.Sequential(
								nn.Linear(input_dim, hidden_dim),
		                        nn.BatchNorm1d(hidden_dim),
		                        nn.ReLU(True),
		                        nn.Dropout(p=dropout)
		                   	) for _ in range(self.seq_len)
						])

		if num_layers >= 2:
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

		self.st_output = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.seq_len)])

	def forward(self, x):
		x = self.input_embed(x)
		out = []
		for i in range(self.seq_len):
			y = self.st_input[i](x)	# input layer
			for hidden in self.st_hidden[i]:
				y = hidden(y)		# hidden layer
			y = self.st_output[i](y)	# output layer
			out.append(y)
		out = torch.cat(out, dim=1)

		return torch.sigmoid(out) if self.sigmoid else out

class MTBinaryPredictor(nn.Module):
	def __init__(self,
				 seq_len=10,
				 num_layers=10,
				 input_dim=16,
				 hidden_dim=1024,
				 dropout=.0,
				 sigmoid=True,
				 pertask_filter=False,
				 lossdrop=False,
				 p_lossdrop=.0,
				 residual=False):

		super(MTBinaryPredictor, self).__init__()
		self.seq_len = seq_len
		self.num_layers = num_layers
		self.sigmoid = sigmoid
		self.pertask_filter = pertask_filter
		self.lossdrop = lossdrop
		self.p_lossdrop = p_lossdrop
		self.residual = residual

		self.input_embed = nn.Embedding(num_embeddings=2**self.seq_len, embedding_dim=input_dim)

		if pertask_filter:
			self.weight_filters = [nn.ParameterList([nn.Parameter(torch.randn(1), requires_grad=True) for _ in range(self.seq_len)]) for _ in range(num_layers)]
			self.bias_filters = [nn.ParameterList([nn.Parameter(torch.randn(1), requires_grad=True) for _ in range(self.seq_len)]) for _ in range(num_layers)]

		self.mt_input = nn.Sequential(
							nn.Linear(input_dim, hidden_dim),
		                    nn.BatchNorm1d(hidden_dim),
		                    nn.ReLU(True),
		                    nn.Dropout(p=dropout)
		               	)

		if self.num_layers >= 2:
			self.mt_hidden = nn.ModuleList([
							  nn.Sequential(
							  	  nn.Linear(hidden_dim, hidden_dim),
								  nn.BatchNorm1d(hidden_dim),
								  nn.ReLU(True),
								  nn.Dropout(p=dropout)
							  ) for _ in range(self.num_layers-2)
						  ])

		if self.pertask_filter:
			self.mt_output = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.seq_len)])
		else:
			self.mt_output = nn.Linear(hidden_dim, self.seq_len)

	def forward(self, x):
		device = x.device
		x = self.input_embed(x) # 512x16

		if self.pertask_filter:
			z = []
			for j in range(self.seq_len):
				# input layer
				y = torch.mm(x, (self.mt_input[0].weight + self.weight_filters[0][j].to(device)).t()) + (self.mt_input[0].bias + self.bias_filters[0][j].to(device))
				y = nn.Sequential(*list(self.mt_input.children())[1:])(y)
				res = y
				# print(self.weight_filters[0][j].requires_grad)

				# hidden layers
				for i in range(self.num_layers-2):
					y = torch.mm(y, (self.mt_hidden[i][0].weight + self.weight_filters[i+1][j].to(device)).t()) + (self.mt_hidden[i][0].bias + self.bias_filters[i+1][j].to(device))
					y = nn.Sequential(*list(self.mt_hidden[i].children())[1:])(y)

				# output layer
				if self.residual:
					y = y + res
				y = torch.mm(y, (self.mt_output[j].weight + self.weight_filters[-1][j].to(device)).t()) + (self.mt_output[j].bias + self.bias_filters[-1][j].to(device))

				z.append(y)

			out = torch.cat(z, dim=1) # batch_size x seq_len
		else:
			# z = []
			# for j in range(self.seq_len):
			# 	# input layer
			# 	y = torch.mm(x, self.mt_input[0].weight.t()) + self.mt_input[0].bias
			# 	y = nn.Sequential(*list(self.mt_input.children())[1:])(y)

			# 	# hidden layers
			# 	for i in range(self.num_layers-2):
			# 		y = torch.mm(y, self.mt_hidden[i][0].weight.t()) + self.mt_hidden[i][0].bias
			# 		y = nn.Sequential(*list(self.mt_hidden[i].children())[1:])(y)

			# 	# output layer
			# 	y = torch.mm(y, self.mt_output[j].weight.t()) + self.mt_output[j].bias

			# 	z.append(y)

			# out = torch.cat(z, dim=1) # batch_size x seq_len

			y = self.mt_input(x)
			res = y
			for hidden in self.mt_hidden:
				y = hidden(y)
			if self.residual:
				y = y + res
			out = self.mt_output(y)

		if self.lossdrop:
			out = F.dropout(out, p=self.p_lossdrop, training=self.training)

		return torch.sigmoid(out) if self.sigmoid else out

if __name__ == '__main__':
	def get_arguments():
		parser = argparse.ArgumentParser(description='Binary Predictor')
		parser.add_argument('--mode', type=str, help='single task (st); multi task (mt)', required=True)
		parser.add_argument('--filter', action='store_true', help='use per-task filter', default=False)
		parser.add_argument('--lossdrop', action='store_true', help='use loss-dropout', default=False)
		parser.add_argument('--residual', action='store_true', help='use residual connection', default=False)
		parser.add_argument('--p_lossdrop', type=float, help='percentage of loss-dropout', default=0.3)
		parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
		parser.add_argument('--epoch', type=int, help='the number of epochs', default=1000)
		parser.add_argument('--batch', type=int, help='batch size', default=128)
		parser.add_argument('--len', type=int, help='sequence length (i.e., x-bit binary)', default=10)
		parser.add_argument('--layers', type=int, help='the number of layers', default=2)
		parser.add_argument('--in_dim', type=int, help='size of input dimension', default=64)
		parser.add_argument('--hid_dim', type=int, help='size of hidden dimension', default=1024)
		parser.add_argument('--seed', type=int, help='random seed', default=0)
		args = parser.parse_args()
		return args

	args = get_arguments()
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	cudnn.deterministic = True

	num_seq=2**args.len
	logger = setup_logger(name=args.mode, save_dir='logs', filename='{}_binary_predictor_{}.txt'.format(get_timestamp(), args.mode))
	logger = logging.getLogger(args.mode)
	
	if args.mode == 'st':
		model = STBinaryPredictor(seq_len=args.len, num_layers=args.layers, input_dim=args.in_dim,
							hidden_dim=args.hid_dim, dropout=0., sigmoid=True)
	elif args.mode == 'mt':
		model = MTBinaryPredictor(seq_len=args.len, num_layers=args.layers, input_dim=args.in_dim,
								hidden_dim=args.hid_dim, dropout=0., sigmoid=True,
								pertask_filter=args.filter, lossdrop=args.lossdrop, p_lossdrop=args.p_lossdrop,
								residual=args.residual)
	input_nums = torch.arange(num_seq)#.float().reshape(-1, 1)
	target_nums = random_number_generator(seq_len=args.len)

	dataloader = DataLoader(TensorDataset(input_nums, target_nums), batch_size=args.batch, shuffle=True, num_workers=8)

	criterion = nn.L1Loss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
	
	model = nn.DataParallel(model, device_ids=[0,1,2,3])
	model = model.cuda()
	model.train()
	# print(model)

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