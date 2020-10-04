# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 09:12:00 2020

@author: rosaz
"""

import torch
import numpy as np
from utils.calculate_time import Timer
from torch.nn import functional as F
from prova import addValueJsonModel
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score

def test_siamese_roc(model, loader_train, loader_valid, directory,version):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    
    timer = Timer()
    
    loader={'train':loader_train, 'valid': loader_valid}
    modalita = ['train','valid']
    
    
    for mode in ['train','valid']:
        print("Modalita ",mode)
        gt = []
        distanze = []
        for  i, batch in enumerate(loader[mode]):
            I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
            #img1, img2, label12, label1, label2
            #l'implementazione della rete siamese è banale:
            #eseguiamo la embedding net sui due input
                    
            phi_i = model(I_i)#img 1
            phi_j = model(I_j)#img2
                    
            print("Output train img1", phi_i.size())
            print("Output train img2", phi_j.size())
            print("Etichetta reale",l_ij)
            labs = l_ij.to('cpu')  
          
        
            dist = F.pairwise_distance(phi_i, phi_j)
            dist = dist.cpu()
            dist = dist.tolist()
            print("DISTANZE ",dist)
        
            gt.extend(list(labs))
            distanze.extend(list(dist))
            
        print("Modalita: "+mode)
        
        print("Curve ROC")
        fpr, tpr, thresholds = roc_curve(gt, distanze)
        
        
        plot_roc(directory,version,fpr,tpr,mode)

        print("Scelta della buona soglia")
        score = tpr+1-fpr
        
        soglia_ottimale = plot_threshold(directory,version,thresholds,score,mode)
        
        
        print("Performance..."+mode)
        
        
        predette = distanze>soglia_ottimale
        
        cm = confusion_matrix(gt, predette)
        #cm=cm/cm.sum(1).reshape(-1,1)
        
        tnr, fpr, fnr, tpr = cm.ravel()
        print("False Positive Rate: {:0.2f}".format(fpr))
        print("True Positive Rate: {:0.2f}".format(tpr))
        
        accuracy = accuracy_score(gt, predette)
        precision = precision_score(gt, predette)
        recall = recall_score(gt, predette)
        f1 = f1_score(gt,predette)
        
        print("Precision: {:0.2f}, Recall: {:0.2f}".format(precision,recall))
        print("Accuracy: {:0.2f} ".format(precision,recall))
        print("F1 score: {:0.2f}".format(f1.mean()))
        
        key=["threshold","accuracy","precision","recall","mf1_score"]
        entry=["threshold_"+mode,"accuracy_"+mode,"precision_"+mode,"recall_"+mode,"f1_score_"+mode]
        value=[soglia_ottimale,accuracy,precision,recall,f1]
        for i in range(5):
            addValueJsonModel(directory+"\\"+"modelTrained.json",version, key[i] ,entry[i],value[i])
        
        key = "performance_test"
        entry=["TNR","FPR","FNR","TPR"]
        value=[tnr, fpr, fnr, tpr]
        addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[0], value[0])
        addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[1], value[1])
        addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[2], value[2])
        addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[3], value[3])
        

def plot_roc(directory, version,fpr,tpr,mod):
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr)
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve '+mod)
    plt.legend(['Distanze '+mod])
    plt.grid()
    plt.savefig(directory+"\\"+version+"\\"+'plotCurve_ROC_'+mod+'.png')
    plt.clf()

def plot_threshold(directory,version,threshold,score,mod):
    plt.figure(figsize=(8,6))
    plt.plot(threshold,score)    
    plt.xlabel('Thresholds')
    plt.ylabel('TPR+(1-FPR)')
    plt.grid()
    optimal_threshold = threshold[np.argmax(score)]
    print(mod+" - La soglia migliore è %0.2f" % optimal_threshold)
    plt.title(mod+" - Optimal threshold: "+str(optimal_threshold))
    plt.savefig(directory+"\\"+version+"\\"+'plotThreshold_'+mod+'.png')
    plt.clf()
    return optimal_threshold
    
            
                    