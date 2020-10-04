# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:11:39 2020

@author: rosaz
"""
from torch.utils.tensorboard import SummaryWriter
import torch
from ContrastiveLoss import ContrastiveLoss 
from os.path import join
from torch.nn import functional as F
import numpy as np
from utils.calculate_time import Timer

from torch import nn
class SoftmaxRegressor(nn.Module):
    def __init__(self, in_features, out_classes):
        """Costruisce un regressore softmax.
        Input:
            in_features: numero di feature in input (es. 4)
            out_classes: numero di classi in uscita (es. 3)"""
        super(SoftmaxRegressor, self).__init__() #richiamo il costruttore della superclasse
        #questo passo è necessario per abilitare alcuni meccanismi automatici dei moduli di PyTorch

        self.linear = nn.Linear(in_features,out_classes) #il regressore softmax restituisce
        #distribuzioni di probabilità, quindi il numero di feature di output coincide con il numero di classi

    def forward(self,x):
        """Definisce come processare l'input x"""
        scores = self.linear(x)
        return scores
    
