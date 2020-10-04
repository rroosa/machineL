# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:41:22 2020

@author: rosaz
"""
import torch
import numpy as np
from utils.calculate_time import Timer

def test_classifier(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    predictions, labels = [], []
    
    timer = Timer()
    for batch in loader:
        x = batch['image'].to(device)
        y = batch['label'].to(device)
        output = model(x)
        preds = output.to('cpu').max(1)[1].numpy()
        labs = y.to('cpu').numpy()
        print("Reali",labs)
        print("Predette",preds)
        predictions.extend(list(preds))
        labels.extend(list(labs))
    
    f='{:.7f}'.format(timer.stop())
    return f, np.array(predictions), np.array(labels)