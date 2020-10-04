# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:31:00 2020

@author: rosaz
"""
from torchvision import transforms
from utils.constants import workdir_dataset, image_path, seed, workdir
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import errno
import json
import random
import torch.nn.functional as nnf

from torch.nn import functional as F
from CSVImageDataset import CSVImageDataset
from PairMoney import PairMoney
import torch
from prova import writeJsonMargin, writeJson, createFolder, readNorm, norm, addKeyValue, addKey,writeJsonNorma
from utils.calculate_time import Timer

class DataSetPairCreate():
    
    def __init__(self,resize=None):
        random.seed(seed)
        np.random.seed(seed)
        
        self.path="Dataset"

        self.resize = resize
        
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_train =CSVImageDataset(workdir,'Dataset/train.csv' ,transform=transform)
        self.dataset_valid =CSVImageDataset(workdir,'Dataset/valid.csv',transform=transform)
        self.dataset_test = CSVImageDataset(workdir,'Dataset/test.csv',transform=transform)
        sample = self.dataset_train[10]
        print(sample['image'].shape)
        print(sample['label'])
        
        #self.controlNormalize(self.path)
        
        #self.run()
    
    def controlMargin(self):
        #controlla se è presente la directory, altrimenti la crea 
        createFolder(self.path)
        #print("Controll")
        if not os.path.exists(self.path+'\\dataSetJson.json'):
            print("1) Calculate mean margine")
            self.runMargin()
            
        else: # se il file esiste controlla se ci sono le key mean e dev 
            res =str(self.resize)
            try:
                with open(self.path+"\\dataSetJson.json","r") as json_file:
                    data = json.load(json_file)
                print(self.path+"\\dataSetJson.json")
                if not (data.get('margineMean_'+res) is None):
                    #print("Qui")
                    margin = data['margineMean_'+res]
                                           
                    response = input("Do you want to re-calculate margin? y | n : ")
                    if(response =="y"):
                        print("recalculate")
                        self.runMargin()
                    elif (response =="n"):
                            print("bypass this step")
                            self.margin = margin
                             
    
                    else:
                            self.controlMargin()
                    
                else:
                    self.runMargin()
            except:
                # se il parsing è errato ricalcola la media e 
                 sys.stderr.write("Error parsing")
                 exit(0)
    
    
        
    
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
                        
    
    def normalizeDataSet(self):
        print("2) Normalize dataSet")
        if self.resize is not None:
            print(self.mean)
            print(self.dev_std)
            self.transform = transforms.Compose([transforms.Resize(self.resize), transforms.ToTensor(), transforms.Normalize(self.mean,self.dev_std)])
        
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean,self.dev_std)])
            
        self.dataset_train_norm = CSVImageDataset(workdir,'Dataset/train.csv' ,transform=self.transform)
        self.dataset_valid_norm = CSVImageDataset(workdir,'Dataset/valid.csv',transform=self.transform)
        self.dataset_test_norm =  CSVImageDataset(workdir,'Dataset/test.csv',transform=self.transform)
        
        print("Sample after the normalization")
        print(self.dataset_train_norm[0]['image'].shape)
        print(self.dataset_train_norm[0]['label'])
        print("immagine normalizza")
        print(self.dataset_train_norm[0])
        
        print("3) Create Dataset of Pair")
        self.pair_money_train = PairMoney('Dataset/','Dataset/train.csv',self.dataset_train_norm,"train")
        numSimilTrain = self.pair_money_train.numSimil
        numDissimilTrain = self.pair_money_train.numDissimil
         #if self.resize is not None:
            #self.controlMargin()
        self.pair_money_test = PairMoney('Dataset/','Dataset/test.csv',self.dataset_test_norm,"test")
        numSimilTest = self.pair_money_test.numSimil
        numDissimilTest = self.pair_money_test.numDissimil
        
        self.pair_money_val = PairMoney('Dataset/','Dataset/valid.csv',self.dataset_valid_norm,"validation")
        numSimilValid = self.pair_money_val.numSimil
        numDissimilValid = self.pair_money_val.numDissimil
        
        value = []
        objTrain = {"splitPair":"train", "numSimil":numSimilTrain,"numDissimil":numDissimilTrain}
        value.append(objTrain)
        objValid = {"splitPair":"validation","numSimil":numSimilValid,"numDissimil":numDissimilValid}
        value.append(objValid)
        objTest = {"splitPair":"test","numSimil": numSimilTest, "numDissimil": numDissimilTest}
        value.append(objTest)
        addKeyValue("Dataset\dataSetJson.json", "dataSetPair", value)
        
        print("4) Show example pair")
        
        plt.figure(figsize=(18,4))
        for ii,i in enumerate(np.random.choice(range(len(self.pair_money_train)),6)):
            plt.subplot(2,10,ii+1)
            plt.title('Pair label:'+str(self.pair_money_train[i][2]))
            plt.imshow(norm(self.pair_money_train[i][0].numpy().transpose(1,2,0)))
            plt.subplot(2,10,ii+11)
            plt.imshow(norm(self.pair_money_train[i][1].numpy().transpose(1,2,0)))
        plt.show()
        
        
        
        #return
        
    
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

    def runMargin(self):
        print(" 1.1) tu...Calculate margin mean on dataset_train")
        
        timer = Timer()    
        somma = 0
        
        
        for i, sample in enumerate (self.pair_money_train):
            #print(i)
            #print("\n")
            I_i,I_j, _ , _ , _ = sample
            #print(len(sample))
            #trasforma in vettori
            #print(type(I_i))
            
            img1 = I_i.view(I_i.size()[0],-1)
            img2 = I_j.view(I_j.size()[0],-1)
            d = F.pairwise_distance(img1, img2)
            #print(img1.size())
 
            somma = somma + d
        
        print(len(self.pair_money_train))
        media = somma/len(self.pair_money_train)
        self.f='{:.7f} sec'.format(timer.stop())
        media = media.numpy()
        self.media = media.tolist()
        
        print("Time calculate margine mean =",self.f)
        print("Mean margine= " ,self.media)
                
        #createFolder(self.model)
        #writeJson(self.model, self.idModel, self.m, self.s, self.f)
        res = str(self.resize)
        value = {"margine":self.media,"timeComputing":self.f}
        
        key = "margineMean_"+res
        addKey("Dataset\dataSetJson.json",key,value)
        
        print("\n")
        
        
        
              
        
    
    

        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        

            
        
        
        