# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:51:25 2020

@author: rosaz
"""

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


def test_siamese_diff(siamese_model, pair_money_test_loader, margine=None):
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
        
        
        res=torch.abs(phi_i.cuda() - phi_j.cuda())
        print("Tipo",type(res))
        print("DIm",res.size())
        labs = l_ij.to('cpu')
                    
        lab_batch =[]
        for  i, el  in enumerate (res):
            print("Tipo",type(el))
            print("DIm",el.size())
                        
            result, indice=torch.max(el,0)
            indice= indice.item()
            print("PREDETTA di un smaple",indice)
            label = l_ij[i].to('cpu')
            label= label.item()
                      
            print(" REALE di un sample",label)
                    
            if label == indice:
                print("Corretta R - P",label,indice)
                            
            else:
                print("Scorretta R - P",label,indice)
                        
            lab_batch.append(indice)
                          
        prediction_test.extend(lab_batch)
        labels_test.extend(list(labs.numpy()))
    #print(type(prediction))
    f='{:.7f}'.format(timer.stop())
    print("dim pred",len(prediction_test))
    print("dim labels",len(labels_test))
    return np.array(prediction_test), np.array(labels_test), f
        
        
        
        
        
        
        
        
    
   
        
        
    
    
    
