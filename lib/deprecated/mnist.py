import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class MNISTDataLoader(Dataset):
	def __init__(self, root, phase, transform):
		assert phase in ['train', 'val', 'test']
		self.phase = phase
		self.transform = transform

		with open(root,'rb') as f :
		    u = pickle._Unpickler(f)
		    u.encoding = 'latin1'
		    train_data, val_data, test_data = u.load()
		    
		    self.dataset = {'train': train_data, 
					     	'val': val_data, 
					     	'test': test_data} 

		inputs, targets = self.dataset[phase]
		inputs = torch.FloatTensor(inputs) 
		targets = torch.LongTensor(targets) 
		data_size = inputs.size(0) 

		inputs = [inputs[i] for i in range(data_size)] 
		
		targets_all = [targets[i] for i in range(data_size)] 
		targets1 = targets[:, 0] # N
		targets2 = targets[:, 1] # N
		targets1 = [targets1[i] for i in range(data_size)] 
		targets2 = [targets2[i] for i in range(data_size)] 

		self.inputs = inputs 
		self.targets1 = targets1 
		self.targets2 = targets2

	def __len__(self):
		return self.dataset[self.phase][0].shape[0]

	def __getitem__(self, idx):
		inputs, targets =  self.inputs[idx], self.self.targets[idx]
		inputs = Image.fromarray(inputs)

		if self.transform is not None:
			img = self.transform(img)

		return inputs, targets