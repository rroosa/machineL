# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:50:11 2020

@author: rosaz
"""

import torch
from torch.nn import functional as F
class ContrastiveLossNorm(torch.nn.Module):
    def __init__(self, m=0.5):
        super(ContrastiveLossNorm, self).__init__()
        self.m = 0.5
        
    def forward(self, phi_i, phi_j, l_ij):
        d = F.pairwise_distance(phi_i, phi_j)
        print(d)
        print(type(d))
        print(d.size())
        
        d_norm = 2.0*( 1.0/(1.0 + torch.exp(-1*d)) - 0.5)
        print(d_norm)
        l = 0.5 * (1 - l_ij.float()) * torch.pow(d_norm,2) + 0.5 * (l_ij.float()) *  torch.pow(torch.clamp( self.m - d_norm, min = 0),2) 
        print("Media loss",(l.mean()).item())
        return l.mean()
