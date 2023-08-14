import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from .coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
from .layers import *

class CoaT_UT(nn.Module):
    def __init__(self, pre=None, arch='medium', num_classes=1, ps=0, num_layers=2, **kwargs):
        super().__init__()
        if arch == 'mini': 
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'small': 
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'medium': 
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128,256,320,512]
        else: raise Exception('Unknown model') 
        
        if pre is not None:
            sd = torch.load(pre)['model']
            print(self.enc.load_state_dict(sd,strict=False))
        
        self.mixer = nn.ModuleList([Tmixer(nc[-2],num_layers=num_layers),
                                    Tmixer(nc[-1],num_layers=num_layers)])
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
        encs = [encs[k] for k in encs]
        encs = [encs[0].view(-1,nt,*encs[0].shape[1:])[:,-1], 
                encs[1].view(-1,nt,*encs[1].shape[1:])[:,-1], 
                self.mixer[-2](encs[2].view(-1,nt,*encs[2].shape[1:])),
                self.mixer[-1](encs[3].view(-1,nt,*encs[3].shape[1:]))]
        dec4 = encs[-1]
        dec3 = self.dec4(dec4,encs[-2])
        dec2 = self.dec3(dec3,encs[-3])
        dec1 = self.dec2(dec2,encs[-4])
        x = self.fpn([dec4, dec3, dec2], dec1)
        x = self.final_conv(self.drop(x))
        if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')
        return x


