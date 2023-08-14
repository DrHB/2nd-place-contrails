import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from .nextvit import nextvit_base, nextvit_large
from .layers import *

class NeXtViT_ULSTM(nn.Module):
    def __init__(self, pre=None, arch='large', num_classes=1, ps=0, 
                 **kwargs):
        super().__init__()
        in_chans = 3
        nc = [96,256,512,1024]
        if arch == 'base': 
            self.enc = nextvit_base()
        elif arch == 'large': 
            self.enc = nextvit_large()
        else: raise Exception('Unknown model') 
        
        if pre is not None:
            sd0 = torch.load(pre)['state_dict']
            sd = OrderedDict()
            for k in sd0:
                if 'backbone.' not in k: continue
                sd[k.replace('backbone.','')] = sd0[k]
            self.enc.load_state_dict(sd, strict=False)
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, num_classes, blur=True))
        self.up_result=1
    
    def forward(self, x):
        x = x[:,:,:5].contiguous()
        nt = x.shape[2]
        x = x.permute(0,2,1,3,4).flatten(0,1)
        x = F.interpolate(x,scale_factor=2,mode='bicubic').clip(0,1)
        encs = self.enc(x)
        #encs = [encs[k] for k in encs]
        encs = [encs[0].view(-1,nt,*encs[0].shape[1:])[:,-1], 
                encs[1].view(-1,nt,*encs[1].shape[1:])[:,-1], 
                self.lstm[-2](encs[2].view(-1,nt,*encs[2].shape[1:]))[:,-1],
                self.lstm[-1](encs[3].view(-1,nt,*encs[3].shape[1:]))[:,-1]]
        dec4 = encs[-1]
        dec3 = self.dec4(dec4,encs[-2])
        dec2 = self.dec3(dec3,encs[-3])
        dec1 = self.dec2(dec2,encs[-4])
        x = self.fpn([dec4, dec3, dec2], dec1)
        x = self.final_conv(self.drop(x))
        if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')
        return x

