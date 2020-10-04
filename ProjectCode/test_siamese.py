# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:36:54 2020

@author: rosaz
"""
from torch.utils.tensorboard import SummaryWriter
import torch
from ContrastiveLoss import ContrastiveLoss 
from os.path import join
from torch.nn import functional as F
import numpy as np
from utils.calculate_time import Timer


def modeLossNorm(m, d ):
    labels = []
    
    d_n = 2.0*( 1.0/(1.0 + torch.exp(-1*d)) - 0.5)
    print("distanze normalizzate", d_n)
    
    for el in d_n:
        if el <= 0.5: # SIMILI
            
            labels.append(0)
        else: # DISSIMILI
            
            labels.append(1)
    return labels

def modeLossTrad(m, d ):
    labels = []
    print("distanze", d)
    for el in d:
        if el <= m: # SIMILI
            
            labels.append(0)
        else: # DISSIMILI
            
            labels.append(1)
    return labels
    
def modeLossSoglia(m, d ):
    labels = []
    
    
    print("distanze con soglia", d)
    
    for el in d:
        if el <= 0.5: # SIMILI
            
            labels.append(0)
        else: # DISSIMILI
            
            labels.append(1)
    return labels

def modeLossDouble(m1,m2, d):
    labels = []
    
    d_n = 2.0*( 1.0/(1.0 + torch.exp(-1*d)) - 0.5)
    print("distanze normalizzate", d_n)
    for el in d_n:
        if el <= m1: # SIMILI
            
            labels.append(0)
        elif el >= m2: # DISSIMILI
            labels.append(1)
    return labels


def test_siamese(siamese_model, pair_money_test_loader, margine=None):
        
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prediction_test = []
    labels_test = []

    timer = Timer()  
    for i, batch in enumerate (pair_money_test_loader):
        print(i)
        I_i, I_j, l_ij, _, _ = [b for b in batch]

        print(len(batch[0]))
        phi_i = siamese_model(I_i)#img 1 # output: rappresentation
        phi_j = siamese_model(I_j)#img2 # output: rapresentation
        print("Rappresenation",type(phi_j))
        d = F.pairwise_distance(phi_i.to('cpu'), phi_j.to('cpu'))


        labs = l_ij.to('cpu')
        print(len(labs))
        #tensor = torch.clamp( margin-d, min = 0) # sceglie il massimo  # sceglie il massimo -- se è zero allora sono dissimili
        #print("max",type(tensor))
        print("size max  ",d.size())
        #print("distanze", d)
        
        if not margine is None:
            if margine == "double":
                listaEtichette = modeLossDouble(0.3, 0.7 ,d)
                
            elif margine =="norm":
                listaEtichette = modeLossNorm(0.5 ,d)
            
            elif margine =="soglia":
                listaEtichette = modeLossSoglia(0.5, d)
                
        else:
             listaEtichette = modeLossTrad(0.8, d)
                    

        prediction_test.extend(listaEtichette)
                
                
        labels_test.extend(list(labs.numpy()))
        print(len(labels_test))
        print(len(prediction_test))

     
    #print(type(prediction))
    f='{:.7f}'.format(timer.stop())
    print("dim pred",len(prediction_test))
    print("dim labels",len(labels_test))
    
    return np.array(prediction_test), np.array(labels_test), f
        
        
        
        
        
        
        
        
    
   
        
        
    
    
    