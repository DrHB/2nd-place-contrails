import torch
from torch import nn
from tqdm import tqdm
import os, random, gc
import numpy as np
from .fastai_fix import *
from .radam import Over9000


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def WrapperAdamW(param_groups, **kwargs):
    return OptimWrapper(param_groups, torch.optim.AdamW)


def WrapperOver9000(param_groups, **kwargs):
    return OptimWrapper(param_groups, opt=Over9000)


class F_th(Metric):
    def __init__(self, ths=np.arange(0.1,0.9,0.01), beta=1): 
        self.ths = ths
        self.beta = beta
        
    def reset(self): 
        self.tp = torch.zeros(len(self.ths))
        self.fp = torch.zeros(len(self.ths))
        self.fn = torch.zeros(len(self.ths))
        
    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred.float()), 
                                  (learn.y > 0.5).float())
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.tp[i] += (p*targ).float().sum().item()
            self.fp[i] += (p*(1-targ)).float().sum().item()
            self.fn[i] += ((1-p)*targ).float().sum().item()

    @property
    def value(self):
        self.dices = (1 + self.beta**2)*self.tp/\
            ((1 + self.beta**2)*self.tp + self.beta**2*self.fn + self.fp + 1e-6)
        return self.dices.max()