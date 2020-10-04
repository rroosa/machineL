# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:23:57 2020

@author: rosaz
"""
import torch

from torch.nn import functional as F
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()
        self.m = m
        print("margin",self.m)
    def forward(self, phi_i, phi_j, l_ij):
        d = F.cosine_similarity(phi_i, phi_j)
        
        l = 0.5 * (1 - l_ij.float()) * torch.pow(d,2) + 0.5 * (l_ij.float()) * torch.pow( torch.clamp( self.m - d, min = 0.0) , 2)
        print("Media loss",(l.mean()).item())
        return l.mean()