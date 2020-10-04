# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:43:33 2020

@author: rosaz
"""
import torch
import numpy as np
from utils.calculate_time import Timer
from torch.nn import functional as F

def test_margine_dynamik(model, loader,soglia, margine=None):
    
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    
    timer = Timer()
    
    for  batch in loader:
        I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
        #img1, img2, label12, label1, label2
        #l'implementazione della rete siamese è banale:
        #eseguiamo la embedding net sui due input
                    
        phi_i = model(I_i)#img 1
        phi_j = model(I_j)#img2
                    
        print("Output train img1", phi_i.size())
        print("Output train img2", phi_j.size())
        print("Etichetta reale",l_ij)
                    
        #l_ij = l_ij.type(torch.LongTensor).to(device)
        labs = l_ij.to('cpu').numpy()
         #distanza euclidea
        euclidean_distance = F.pairwise_distance(phi_i, phi_j)
        testing_label = euclidean_distance > soglia # 0 if same, 1 if not same (progression) 
        #equals = training_label.int() == l_ij.int() # 1 if true
        
        testing_label = testing_label.int().numpy()
        
        print("Reali",labs)
        print("Predette",testing_label)
        predictions.extend(list(testing_label))
        labels.extend(list(labs))
    
        print("lunghezza predette",len(predictions))
        print("lunghezza reali",len(labels))
        
    f='{:.7f}'.format(timer.stop())
    return f, np.array(predictions), np.array(labels)