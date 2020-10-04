# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:22:57 2020

@author: rosaz
"""

import torch
import numpy as np
from utils.calculate_time import Timer
from torch.nn import functional as F

def test_siamese_margine(model, loader, soglia):
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
                    
        dist = F.pairwise_distance(phi_i, phi_j)
        dist = dist.cpu()
        dist = dist.tolist()
        print("DISTANZE ",dist)
                    
        pred = []
        label = []
        labs = l_ij.to('cpu')
        for j in dist:
            if j<= soglia:
                                
                pred.append(0)
                                
            else:
                pred.append(1)

        label.extend(list(labs.numpy()))
        print("Reali del batch ",label)
        print("Predette del batch",pred)
        
        predictions.extend(list(pred))
        labels.extend(list(label))
    
        print("lunghezza predette",len(predictions))
        print("lunghezza reali",len(labels))
        
    f='{:.7f}'.format(timer.stop())
    return f, np.array(predictions), np.array(labels)