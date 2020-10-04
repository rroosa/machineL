# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:26:50 2020

@author: rosaz
"""

import torch
import math
from torch.nn import functional as F
class ContrastiveLossDouble(torch.nn.Module):
    def __init__(self, m1=0.8, m2 = 1.2):
        super(ContrastiveLossDouble, self).__init__()
        self.m1 = m1
        self.m2 = m2
        print("Margin1",m1)
        print("MArgin2",m2)
        
    def forward(self, phi_i, phi_j, l_ij):
        
        d = F.pairwise_distance(phi_i, phi_j)
        
        l = 0.5 * (1 - l_ij.float()) *  torch.pow(torch.clamp( d - self.m1 , min = 0),2)  + \
            0.5 * (l_ij.float()) *  torch.pow(torch.clamp( self.m2 - d, min = 0),2) 
        print("Media loss",(l.mean()).item())
        return l.mean()
