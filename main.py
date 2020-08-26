import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lib.config import cfg
from lib.trainer import Trainer
from lib.utils.logger import setup_logger
from lib.utils.miscellaneous import get_timestamp, save_config

def get_arguments():
	parser = argparse.ArgumentParser(description='Multitask Learning (CostOut, Filter)')
	parser.add_argument('--gpu', type=str, help='0; 0,1; 0,3; etc', required=True)
	parser.add_argument('--mode', type=str, help='baseline; costout; filter', required=True)
	parser.add_argument('--eval', action='store_true', help='evaluate mode')
	parser.add_argument('--resume', action='store_true', help='resume checkpoint')
	parser.add_argument('--cfg', default='configs/base.yaml')
	args = parser.parse_args()
	return args

def main():
	args = get_arguments()
	cfg.merge_from_file(args.cfg)

	if cfg.ETC.SEED is not None:
		random.seed(cfg.ETC.SEED)
		np.random.seed(cfg.ETC.SEED)
		torch.manual_seed(cfg.ETC.SEED)
		cudnn.deterministic = True

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	phase = 'eval' if args.eval else 'train'
	args.save = os.path.join('save', args.mode, cfg.MODEL.BASE_MODEL)
	args.checkpoint = os.path.join(args.save, cfg.MODEL.CHECKPOINT) # 'best_loss.pth.tar'
	if not os.path.exists(args.save):
		os.makedirs(args.save)

	if not os.path.exists('logs'):
		os.mkdir('logs')

	logger = setup_logger(name=args.mode, save_dir='logs',
		filename='{}_{}_{}_{}.txt'.format(get_timestamp(), args.mode, cfg.MODEL.BASE_MODEL, phase))
	logger.info(args)
	logger.info('Loaded configuration file {}'.format(args.cfg))
	output_config_path = os.path.join('logs', 'config.yml')
	logger.info('Saving config into: {}'.format(output_config_path))
	save_config(cfg, output_config_path)

	trainer = Trainer(cfg, args)

	if args.eval:
		trainer.test()
	else:
		trainer.train()

if __name__ == '__main__':
	main()