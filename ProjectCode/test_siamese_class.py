# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:45:21 2020

@author: rosaz
"""
import torch
import numpy as np
from utils.calculate_time import Timer

def test_siamese_class(model, loader, margine=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    
    timer = Timer()
    
    for  batch in loader:
        I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
        #img1, img2, label12, label1, label2
        #l'implementazione della rete siamese è banale:
        #eseguiamo la embedding net sui due input
                    
        phi_i = model(I_i)#img1
        phi_j = model(I_j)#img2
                    
        print("Output train img1", phi_i.size())
        print("Output train img2", phi_j.size())
        print("Etichetta reale",l_ij)
                    
        f = torch.cat((phi_i,phi_j),1)
                    
        output = model.fc2(f)
        preds = output.to('cpu').max(1)[1].numpy()
        labs = l_ij.to('cpu').numpy()
        print("Reali",labs)
        print("Predette",preds)
        predictions.extend(list(preds))
        labels.extend(list(labs))
    
        print("lunghezza predette",len(predictions))
        print("lunghezza reali",len(labels))
        
    f='{:.7f}'.format(timer.stop())
    return f, np.array(predictions), np.array(labels)