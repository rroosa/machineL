# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:13:30 2020

@author: rosaz
"""
from torch.utils.tensorboard import SummaryWriter
import torch
from ContrastiveLoss import ContrastiveLoss 
from os.path import join
from torch.nn import functional as F
import numpy as np
from SoftmaxRegressor import SoftmaxRegressor
from utils.calculate_time import Timer
from torch import nn

def test_classifier(model, loader, in_features):
    device = "cuda" 
    #if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    softmaxregressor = SoftmaxRegressor(in_features,2)
    softmaxregressor.to(device)
    timer = Timer()
    
    for batch in loader:
        
        I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
        #img1, img2, label12, label1, label2
        #l'implementazione della rete siamese è banale:
        #eseguiamo la embedding net sui due input
        print()
        print(len(batch[0]))
        phi_i = model(I_i)#img 1 # output: rappresentation
        phi_j = model(I_j)#img2 # output: rapresentation
        print("Rappresenation",type(phi_j))
        print("Rappresenation",phi_j.shape)
        phi_i = phi_i.to('cpu')
        phi_i=phi_i.detach().numpy()
        phi_j = phi_j.to('cpu')
        phi_j = phi_j.detach().numpy()
        
        """
        #Il calcolo della distanza del coseno viene effettuato prendendo il prodotto punto dei vettori.
        diff = phi_i.dot(phi_j)
       # diff = abs(phi_i-phi_j.numpy())
        print("diff",type(diff))
        print("diff",diff.shape)
        diff = torch.reshape(diff, (-1,))
        print("after diff",type(diff))
        print("after diff",diff.shape)
        
        #d = F.pairwise_distance(phi_i.to('cpu'), phi_j.to('cpu'))
        #l1_layer =Lamba(lambda tensors: )
        #print("Tipo della d",type(d))
        #print("Lunghezza d", d.shape)
        #d =d.to(device)
        
        softmax = softmaxregressor(diff.shape,2)
        output = softmax(diff)
        """
        # crea un vettore dai confronti di vettori positivi e negativi 
        print("Dim phii",phi_i.shape)
        print("Dim pij", phi_j.shape)
        v = np.concatenate ((phi_i, phi_j))
        # porta e alla potenza di ogni valore nel vettore 
        exp = np.exp (v)
        # divide ogni valore per la somma dei valori esponenziali 
        softmax_out = exp / np.sum (exp)
        print("DIm softmax", softmax_out.shape )
        
        #preds = output.to('cpu').max(1)[1].numpy()
        
        labs = l_ij.to('cpu')
        predictions.extend(list(preds))
        labels.extend(list(labs))
    f='{:.7f} sec'.format(timer.stop())
    return np.array(predictions), np.array(labels),f