# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:25:33 2020

@author: rosaz
"""


import torch
import numpy as np
from utils.calculate_time import Timer

def test_classifierPair(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    predictions, labels = [], []
    
    timer = Timer()
    for batch in loader:
        pred = []
        I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
        out_img1 = model(I_i)#etichetta1
        out_img2 = model(I_j)#etichetta2
        
        preds_img_1 = out_img1.to('cpu').max(1)[1].numpy()
        preds_img_2 = out_img2.to('cpu').max(1)[1].numpy()
        
        
        preds_img_1= preds_img_1.tolist()
        
        
        preds_img_2=preds_img_2.tolist()
        
        for j in range(len(preds_img_2)):
            if(preds_img_1[j] <= preds_img_2[j]):
                pred.append(0) 
            else:
                pred.append(1) 
        
        
        
        labs = l_ij.to('cpu').numpy()#reali
        print("Reali ",labs)
        
        labels.extend(list(labs))
        print("lungheza reali",len(labels))
        
        predictions.extend(list(pred))
        print("lunghezza predette",len(predictions))
    
    f='{:.7f}'.format(timer.stop())
    
    return f, np.array(predictions), np.array(labels)