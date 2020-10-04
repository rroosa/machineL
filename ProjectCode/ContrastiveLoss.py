# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:06:36 2020

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
        #distanza euclidea
        d = F.pairwise_distance(phi_i, phi_j)
        
        #contrastive loss single margine
        l = 0.5 * (1 - l_ij.float()) * torch.pow(d,2) + 0.5 * (l_ij.float()) * torch.pow( torch.clamp( self.m - d, min = 0.0) , 2)

        print("Media loss",(l.mean()).item())
        return l.mean()