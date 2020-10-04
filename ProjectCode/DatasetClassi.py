# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:55:31 2020

@author: rosaz
"""
from torchvision import transforms
from utils.constants import workdir_dataset, image_path, seed, workdir
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

import json
import random
import torch.nn.functional as nnf

from torch.nn import functional as F
from CSVImageDataset import CSVImageDataset
from PairMoney import PairMoney
import torch
from prova import writeJsonMargin, writeJson, createFolder, readNorm, norm, addKeyValue, addKey,writeJsonNorma
from utils.calculate_time import Timer
from torchvision import transforms

from utils.constants import workdir_dataset, image_path, seed, workdir
class DatasetClassi():
    
    def __init__(self,resize=None):
        random.seed(seed)
        np.random.seed(seed)
        
        self.path="Dataset"

        self.resize = resize
      
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_train =CSVImageDataset(workdir,'Dataset/train_base.csv' ,transform=transform)
        self.dataset_valid =CSVImageDataset(workdir,'Dataset/valid_base.csv',transform=transform)
        self.dataset_test = CSVImageDataset(workdir,'Dataset/test_base.csv',transform=transform)
        sample = self.dataset_train[10]
        print(sample['image'].shape)
        print(sample['label'])
        
    def controlNormalize(self):
        #controlla se è presente la directory, altrimenti la crea 
        createFolder(self.path)
        #print("Controll")
        if not os.path.exists(self.path+'\\dataSetJson.json'):
            print("1) Checking: mean, dev_std")
            self.run()
            
        else: # se il file esiste controlla se ci sono le key mean e dev 
            
            try:
                with open(self.path+"\\dataSetJson.json","r") as json_file:
                    data = json.load(json_file)
                print(self.path+"\\dataSetJson.json")
                if not (data.get('normalize') is None):
                    #print("Qui")
                    norm = data['normalize']
                    
                    if not (norm.get('mean') and norm.get('dev_std')) is None:
                        
                        response = input("Do you want to re-calculate the mean and standard deviation? y | n : ")
                        if(response =="y"):
                            print("recalculate")
                            self.run()
                        elif (response =="n"):
                            print("bypass this step!!")
                            self.mean = tuple(norm['mean'])
                            print(self.mean)
                            self.dev_std = tuple(norm['dev_std'])
                            print(self.dev_std)
                            self.normalizeDataSet()
                            return
                        else:
                            self.controlNormalize()
                    else:
                        self.run()
                else:
                    self.run()
            except:
                # se il parsing è errato ricalcola la media e 
                 sys.stderr.write("Error parsing")
                 exit(0)


    def run(self):
        print(" 1.1) Calculate mean and dev_std on dataset_train")
        
        timer = Timer()    
        
        m = np.zeros(3)
        for sample in self.dataset_train:
            m+=sample['image'].sum(1).sum(1).numpy() #accumuliamo la somma dei pixel canale per canale

        #dividiamo per il numero di immagini moltiplicato per il numero di pixel
        m=m/(len(self.dataset_train)*300*300)

        #procedura simile per calcolare la deviazione standard
        s = np.zeros(3)
        for sample in self.dataset_train:
            s+=((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1).numpy()

        s=np.sqrt(s/(len(self.dataset_train)*300*300))
        
        self.f='{:.7f} sec'.format(timer.stop())
        
        self.dev_std = s
        self.mean = m
        print(self.f)
        print("Dev.std=",self.dev_std)
        print("Mean= " ,self.mean)
                
        #createFolder(self.model)
        #writeJson(self.model, self.idModel, self.m, self.s, self.f)
        writeJsonNorma("Dataset\dataSetJson.json",self.mean,self.dev_std,self.f)
        
        print("\n")
        
        self.normalizeDataSet()
        
    def normalizeDataSet(self):
        print("2) Normalize dataSet")
        if self.resize is not None:
            print(self.mean)
            print(self.dev_std)
            print("Resize",self.resize)
              
            self.transform = transforms.Compose([transforms.Resize(self.resize),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize(self.mean,self.dev_std)])
        
        else:
            print(self.mean)
            print(self.dev_std)
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean,self.dev_std)])
            
        self.dataset_train_norm = CSVImageDataset(workdir,'Dataset/train_base.csv' ,transform=self.transform)
        self.dataset_valid_norm = CSVImageDataset(workdir,'Dataset/valid_base.csv',transform=self.transform)
        self.dataset_test_norm =  CSVImageDataset(workdir,'Dataset/test_base.csv',transform=self.transform)
        print("Num campioni train", len(self.dataset_train_norm))
        print("Sample after the normalization")
        print(self.dataset_train_norm[0]['image'].shape)
        print(self.dataset_train_norm[0]['label'])
        print("immagine normalizza")
        print(self.dataset_train_norm[0])
            
