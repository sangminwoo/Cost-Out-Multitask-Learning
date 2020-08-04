import os
import torch
import numpy as np

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        self.mean = torch.FloatTensor(self.mean).view(1,3,1,1).expand_as(tensor).to(tensor.device)
        self.std = torch.FloatTensor(self.std).view(1,3,1,1).expand_as(tensor).to(tensor.device)

        return tensor.sub(self.mean).div(self.std)

    def undo(self, tensor):
        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        self.mean = torch.FloatTensor(self.mean).view(1,3,1,1).expand_as(tensor).to(tensor.device)
        self.std = torch.FloatTensor(self.std).view(1,3,1,1).expand_as(tensor).to(tensor.device)

        return tensor.mul(self.mean).add(self.std)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class AverageMeter:
    '''
    Computes and stores the average and current value
    '''
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

class ConvergenceChecker:
    def __init__(self, threshold=1e-4):
        self.threshold = threshold
        self.prev_loss = np.inf

    def check(self, cur_loss):
        if cur_loss == np.inf:
            return False
            
        if self.prev_loss * (1 - self.threshold) < cur_loss < self.prev_loss:
            self.prev_loss = cur_loss
            return True
        else:
            self.prev_loss = cur_loss
            return False

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) # e.g., 32

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True) # 32xk
        pred = pred.t() # kx32
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # e.g., 1x32 -> kx32

        res = []
        for k in topk: # e.g., (1, 5)
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) #
            res.append(correct_k.mul_(100.0 / batch_size))
        return res