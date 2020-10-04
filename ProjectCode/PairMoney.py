# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:33:26 2020

@author: rosaz
"""

import pandas as pd
import torch
import numpy as np
from torch.utils import data
from os.path import join
from PIL import Image

class PairMoney(data.Dataset):
    
    def __init__(self, data_root,dataset_cvs, dataset_img ,mode):
        #self.mnist = MNIST(root = root, train = train, transform = transform, download = download)

        #lista di liste che contiene gli indici elementi appartenenti alle singole classi
        #in pratica class_to_indices[4] contiene gli indici degli elementi di classe 4
        
        self.data_root = data_root
        self.datasetPart = pd.read_csv(dataset_cvs)
        
        self.datasetImg = dataset_img
        self.class_to_indices = [np.where(self.datasetPart['label'] == label)[0] for label in range(5)]
        self.mode = mode
        
        #self.transform = transform
        #genera le coppie
        self.generate_pairs()

    def generate_pairs(self):
        """Genera le coppie, associando ad ogni elemento di datasetPart un nuovo elemento"""
        
        #creiamo un vettore di etichette lij [ (<=) 0 VERO] [(>) 1 FALSO]
        #self.pair_labels = (np.random.rand(len(self.datasetPart))>0.5).astype(int)
        self.pair_labels = []
        
       
        #il primo elmento della coppia i-esima sarà sempre l'elemento i-esimo di datasetPart
        
        #paired_idx conterrà i secondi elementi delle coppie
        self.paired_idx = []
        self.img_1 = []
        self.img_2 = []
        self.label_1 = []
        self.label_2 = []
        
        #
        
        for i in range( len(self.datasetPart) ):
            #print("Elemento ",i)
            #print("etichetta l1",self.datasetPart['label'][i])
            
            #otteniamo la classe del primo elemento della coppia
            c1 = self.datasetPart['label'][i].item()
            #print("Classe img_1",c1)
            
            if c1 == 0:
                #scegli un elemento della classe maggiore o uguale a 0
                index_up = [i for i in range(c1,5)]
                #print(index_up)
                j = np.random.choice(index_up)
                #print("classe img_2",j)
                j = np.random.choice(self.class_to_indices[j])
                #print("Elemento_2: ",j)
                lij = 0
                
                
            
            else: 
                #se la c1>0
            
                #scegli lij in maniera casuale (genera un  numero tra 0 1) e se è > 0.375 (3/8) = 1 altrimenti 0 
                lij = (np.random.rand(1)>0.375).astype(int)
                #print("etichetta",lij)
                lij = lij[0]
                if lij==0:
                    #scegli un elemento di classe maggiore o uguale
                    index_up = [i for i in range(c1,5)]
                    #print(index_up)
                    j = np.random.choice(index_up)
                    #print("classe img_2",j)
                    j = np.random.choice(self.class_to_indices[j])
                    #print("Elemento_2: ",j)
                else:
                    
                    index_low = [i for i in range(0,c1)]
                    #print("Lista",index_low)
                    
                    j = np.random.choice(index_low)
                    #print("classe img_2",j)
                    #campioniamo da quella classe
                    j = np.random.choice(self.class_to_indices[j])
                    #print("Elemento_2: ",j)

                #conserviamo l'indice del secondo elemento della coppia
            self.paired_idx.append(j)
            self.pair_labels.append(lij)
            
            l1 = self.datasetPart['label'][i].item()
            
            l2 = self.datasetPart['label'][j].item()
            
            img1 = self.datasetPart['path'][i]
            img2 = self.datasetPart['path'][j]
            
            self.img_1.append(img1)
            self.img_2.append(img2)
            self.label_1.append(l1)
            self.label_2.append(l2)
            
        
        
        
    
        dataset_part = pd.DataFrame({'path_img_1':self.img_1, 'label_img_1':self.label_1, 'path_img_2':self.img_2, 'label_img_2':self.label_2 , 'lij':self.pair_labels })
        
        dataset_part.to_csv('Dataset/'+'Pair_'+self.mode+'.csv')
        self.numSimil = self.pair_labels.count(0)
        self.numDissimil = self.pair_labels.count(1)
        print(self.numSimil)
        print(self.numDissimil)
            
    
            
            
    def __len__(self):
         #tante coppie quanti sono gli elemnti di datasetPart
        return len(self.datasetPart)

    def __getitem__(self, i):
        #la prima immagine della coppia è l'elemento i-esimo del dataset_img
        campione_1 = self.datasetImg[i]
        campione_2 = self.datasetImg[self.paired_idx[i]] #secondo elemento associato
        l12 = self.pair_labels[i] #etichetta della coppia

        
        #l1 = self.datasetPart['label'][i].item()
        #l2 = self.datasetPart['label'][self.paired_idx[i]].item()
        
        #img1 = self.datasetPart['path'][i]
        #img2 = self.datasetPart['path'][self.paired_idx[i]]
        
        #l = self.pair_labels[i] #etichetta della coppia
        #print(l)
        #restituiamo le due immagini, l'etichetta lij
        #e le etichette delle due immagini
        #return img1, img2, l, l1, l2
        img1= campione_1['image']
        img2= campione_2['image']
        l1= campione_2['label']
        l2= campione_2['label']
        return img1, img2, l12, l1, l2