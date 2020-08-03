import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import Trainer

def get_arguments():
	'''
	train:
		(Baseline) python main.py --gpu 0,1,2,3
		(Costout) python main.py --gpu 0,1,2,3 -costout
	
	evaluate:
		(Baseline) python main.py --gpu 0,1,2,3 --eval
		(Costout) python main.py --gpu 0,1,2,3 --costout --eval
	'''
	parser = argparse.ArgumentParser(description='CostOut (Experimental)')
	# parser.add_argument('-n', '--number', type=int, default=2, help='number of tasks')
	# parser.add_argument('-t', '--task', type=str, default='m10', help='m10: mnist+cifar-10; i100: imagenet+cifar-100')
	parser.add_argument('--costout', action='store_true', help='use costout')
	parser.add_argument('--eval', dest='eval', action='store_true', help='evaluate model')
	parser.add_argument('--resume', action='store_true', help='resume checkpoint')
	parser.add_argument('--gpu', type=str, help='0; 0,1; 0,3; etc', required=True)
	parser.add_argument('--dataset1', type=str, default='cifar-10', help='mnist, cifar-10, cifar-100, imagenet')
	parser.add_argument('--dataset2', type=str, default='cifar-100', help='mnist, cifar-10, cifar-100, imagenet')
	parser.add_argument('--save', type=str, default='save/', help='directory to save checkpoint')
	parser.add_argument('--checkpoint', type=str, default='checkpoint_99.pth.tar', help='model to resume')

	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--epoch', default=100, type=int, metavar='N', help='default: 100')
	parser.add_argument('--model', default='resnet18', type=str, help='mlp, resnet18, 50, 101')
	parser.add_argument('--optimizer', default='SGD', type=str, help='Adam, SGD (default: SGD)')
	parser.add_argument('--lr', type=float, default=1e-3, help='default: 1e-3')
	parser.add_argument('--dropout', type=float, default=0., help='default: 0')
	parser.add_argument('--bn_momentum', type=float, default=1e-2, help='batch normalization momentum (default: 1e-2)')
	parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float, metavar='W', help='default: 1e-5')
	parser.add_argument('--momentum', default=0., type=float, metavar='M', help='momentum (default: 0)')

	parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='data loading workers (default: 4)')
	parser.add_argument('-v', '--verbose', default=True, action='store_true', help='print progress')
	parser.add_argument('--seed', default=0, type=int, help='random seed')
	args = parser.parse_args()
	return args

def main():
	args = get_arguments()

	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	mode = 'costout' if args.costout else 'baseline'
	args.save = os.path.join(args.save, mode, args.model)
	args.checkpoint = os.path.join(args.save, 'checkpoint_99.pth.tar') # 'best_loss.pth.tar'
	if not os.path.exists(args.save):
		os.makedirs(args.save)

	trainer = Trainer(args,
					  epoch=args.epoch,
					  model=args.model,
					  optimizer=args.optimizer,
					  verbose=args.verbose)

	if args.eval:
		trainer.test()
	else:
		trainer.train()

if __name__ == '__main__':
	main()