import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import math
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

class PixelShuffle_ICNR(nn.Sequential):
    def __init__(self, ni, nf=None, scale=2, blur=True):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [nn.Conv2d(ni, nf*(scale**2), 1), LayerNorm2d(nf*(scale**2)), 
                  nn.GELU(), nn.PixelShuffle(scale)]
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)
    
class FPN(nn.Module):
    def __init__(self, input_channels:list, output_channels:list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.GELU(), LayerNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
            for in_ch, out_ch in zip(input_channels, output_channels)])
        
    def forward(self, xs:list, last_layer):
        hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear') 
               for i,(c,x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class UnetBlock(nn.Module):
    def __init__(self, up_in_c:int, x_in_c:int, nf:int=None, blur:bool=False,
                 **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = LayerNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2,32)
        self.conv1 = nn.Sequential(nn.Conv2d(ni, nf, 3, padding=1),nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1),nn.GELU())
        self.relu = nn.GELU()

    def forward(self, up_in:torch.Tensor, left_in:torch.Tensor) -> torch.Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))
    
class UpBlock(nn.Module):
    def __init__(self, up_in_c:int, nf:int=None, blur:bool=True,
                 **kwargs):
        super().__init__()
        ni = up_in_c//4
        self.shuf = PixelShuffle_ICNR(up_in_c, ni, blur=blur, **kwargs)
        nf = nf if nf is not None else max(up_in_c//4,16)
        self.conv = nn.Sequential(nn.Conv2d(ni, ni, 3, padding=1),
                                  LayerNorm2d(ni) if ni >= 16 else nn.Identity(),
                                  nn.GELU(),nn.Conv2d(ni, nf, 1))

    def forward(self, up_in:torch.Tensor) -> torch.Tensor:
        return self.conv(self.shuf(up_in))
    
class LSTM_block(nn.Module):
    def __init__(self, n, bidirectional=False, num_layers=1, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(n, n if not bidirectional else n//2, batch_first=True,
                            bidirectional=bidirectional, num_layers=num_layers)
    
    def forward(self,x):
        s = x.shape
        x = x.flatten(-2,-1).permute(0,3,1,2).flatten(0,1)
        x = self.lstm(x)[0]
        x = x.view(-1,s[3],s[4],s[1],s[2]).permute(0,3,4,1,2)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
#BEiTv2 block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, xq, xk, xv, attn_mask=None, key_padding_mask=None):
        if self.gamma_1 is None:
            x = xq + self.drop_path(self.attn(self.norm1(xq),self.norm1(xk),self.norm1(xv),
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0])
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = xq + self.drop_path(self.gamma_1 * self.attn(self.norm1(xq),self.norm1(xk),self.norm1(xv),
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0])
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class Tmixer(nn.Module):
    def __init__(self, n, head_size=32, num_layers=2, **kwargs):
        super().__init__()
        self.seq_enc = SinusoidalPosEmb(n)
        self.blocks = nn.ModuleList([Block(n,n//64) for i in range(num_layers)])
    
    def forward(self,x):
        B,N,C,H,W = x.shape
        x = x.flatten(-2,-1).permute(0,1,3,2)
        
        enc = self.seq_enc(torch.arange(N, device=x.device)).view(1,N,1,C)
        xq = x[:,-1] + enc[:,-1]
        xk = (x + enc).flatten(1,2)
        xv = x.flatten(1,2)
        
        for m in self.blocks: xq = m(xq,xk,xv)
        
        x = xq.view(B,H,W,C).permute(0,3,1,2)
        return x


