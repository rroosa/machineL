# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:44:21 2020

@author: rosaz
"""


import torch
import math
import statistics
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from timeit import default_timer as timer
from AverageValueMeter import AverageValueMeter
from utils.calculate_time import Timer
from sklearn.metrics import accuracy_score
from torch import nn
from prova import net_save, writeJsonModelEpoca, saveArray,addValueJsonModel
from os.path import join
from matplotlib import pyplot as plt
from torch.nn import functional as F
from scipy.stats import norm
import numpy as np
def gaussian_distribition(directory, version, model, train_loader, valid_loader,test_loader,resize,batch_size, exp_name='model_1'):
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'valid' : valid_loader,
        'test'  : test_loader
        }

    tempo = Timer() 
    start = timer()
    
    array_total_0 = []
    array_total_1 = []
    
    #iteriamo tra due modalità: train, validation e test
    for mode in ['train','valid','test']:
        model.eval()
        for i, batch in enumerate(loader[mode]):
            
            distance_1 = []
            distance_0 = []
            
            print("num batch:", i)
            I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    
            phi_i = model(I_i)#img 1
            phi_j = model(I_j)#img2
            euclidean_distance = F.pairwise_distance(phi_i, phi_j, metric="cosine")  
            
            euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu())
            labs = l_ij.to('cpu').numpy()
            print(euclid_tmp)
            print(labs)
            distance_1 = [distance for distance , label in zip (euclid_tmp, labs) if label == 1]
            distance_0 = [distance for distance, label in zip (euclid_tmp, labs) if label == 0]
            
            print(distance_1)
            print(distance_0)
            if(len(distance_0)!=0):
                array_total_0.extend(distance_0)
            if(len(distance_1)!=0):
                array_total_1.extend(distance_1)
            
            print("len_0:",len(array_total_0))
            print("len_1:", len(array_total_1))
    
    tot_sample = len(array_total_0)+ len(array_total_1)
    print("num tot:", tot_sample)
    
    print("Distribution gaussian_norm")
    

    

    mu_0 = statistics.mean(array_total_0)
    print("Media 0:",mu_0)
    somma = 0
    for i in array_total_0:
        somma = somma + math.pow(i-mu_0,2)
    
    sigma_0 = math.sqrt(somma / len(array_total_0))
    
    print("Dev_std_0:",sigma_0)
    
    # ---------------------------
    mu_1 = statistics.mean(array_total_1)
    print("Media_1:",mu_1)
    somma = 0
    for i in array_total_1:
        somma = somma + math.pow(i-mu_1,2)
    
    sigma_1 = math.sqrt(somma / len(array_total_1))
    
    print("Dev_std_1:",sigma_1)
    
    
    
    g_0 = norm(mu_0,sigma_0)
    g_1 = norm(mu_1,sigma_1)
    x_0=np.linspace(0, max(array_total_0),100)
    x_1=np.linspace(0, max(array_total_1),100)
    plt.figure(figsize=(15,6))
    
    plt.hist(array_total_0, bins=100, density = True)
    plt.hist(array_total_1, bins=100, density = True)

    plt.plot(x_0, g_0.pdf(x_0))
    plt.plot(x_1, g_1.pdf(x_1))
    plt.grid()
    plt.legend(['Densità Stimata_0','Densità Stimata_1','Distribuzione Gaussiana_0','Distribuzione Gaussiana_1'])
    plt.savefig(directory+"\\"+version+"\\"+'plotDistribution_2.png')
    plt.show()
    
            #euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
        
            #training_euclidean_distance_history.extend(euclid_tmp)



def gaussian_distribution_train_margine_single(directory, version, train_loader,resize,batch_size, path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model= torch.load(path)
    model.to(device)

    tempo = Timer() 
    start = timer()
    
    array_total_0 = []
    array_total_1 = []
    

    model.eval()
    for i, batch in enumerate(train_loader):
            
        distance_1 = []
        distance_0 = []
            
        print("num batch:", i)
        I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    
        phi_i = model(I_i)#img 1
        phi_j = model(I_j)#img2
        euclidean_distance = F.pairwise_distance(phi_i, phi_j)  
            
        euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu())
        labs = l_ij.to('cpu').numpy()
        print(euclid_tmp)
        print(labs)
        distance_1 = [distance for distance , label in zip (euclid_tmp, labs) if label == 1]
        distance_0 = [distance for distance, label in zip (euclid_tmp, labs) if label == 0]
            
        print(distance_1)
        print(distance_0)
        if(len(distance_0)!=0):
            array_total_0.extend(distance_0)
        if(len(distance_1)!=0):
            array_total_1.extend(distance_1)
            
        print("len_0:",len(array_total_0))
        print("len_1:", len(array_total_1))
    
    tot_sample = len(array_total_0)+ len(array_total_1)
    print("num tot:", tot_sample)
    
    print("Distribution gaussian_norm")
    

    

    mu_0 = statistics.mean(array_total_0)
    print("Media 0:",mu_0)
    somma = 0
    for i in array_total_0:
        somma = somma + math.pow(i-mu_0,2)
    
    sigma_0 = math.sqrt(somma / len(array_total_0))
    
    print("Dev_std_0:",sigma_0)
    
    # ---------------------------
    mu_1 = statistics.mean(array_total_1)
    print("Media_1:",mu_1)
    somma = 0
    for i in array_total_1:
        somma = somma + math.pow(i-mu_1,2)
    
    sigma_1 = math.sqrt(somma / len(array_total_1))
    
    print("Dev_std_1:",sigma_1)
    
    key="mediaDistrib"
    entry="media_0"
    value=mu_0
    
    entry1="media_1"
    value1=mu_1
    
    
    g_0 = norm(mu_0,sigma_0)
    g_1 = norm(mu_1,sigma_1)
    x_0=np.linspace(0, max(array_total_0),100)
    x_1=np.linspace(0, max(array_total_1),100)
    plt.figure(figsize=(15,6))
    media_0='{:.3f}'.format(mu_0)
    media_1='{:.3f}'.format(mu_1)
    addValueJsonModel(directory+"\\"+"modelTrained.json",version, key ,entry, media_0)
    addValueJsonModel(directory+"\\"+"modelTrained.json",version, key ,entry1, media_1)
    
    plt.hist(array_total_0, bins=100, density = True)
    plt.hist(array_total_1, bins=100, density = True)

    plt.plot(x_0, g_0.pdf(x_0))
    plt.plot(x_1, g_1.pdf(x_1))
    plt.grid()
    plt.title("Media_0: "+media_0+"   Media_1: "+media_1)
    plt.legend(['Densità Stimata_0','Densità Stimata_1','Distribuzione Gaussiana_0','Distribuzione Gaussiana_1'])
    plt.savefig(directory+"\\"+version+"\\"+'plotDistribution_ofClassifacation.png')
    plt.clf()
    #plt.show()