# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:02:38 2020

@author: rosaz
"""

import torch
from torch.utils import data
from os.path import join
from PIL import Image
import pandas as pd
from utils.constants import workdir 

class CSVImageDataset(data.Dataset):
    def __init__(self, data_root, csv, transform = None):
        self.data_root = data_root
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        im_path, im_label = self.data.iloc[i]['path'], self.data.iloc[i].label
        #il dataset contiene alcune immagini in scala di grigi
        #convertiamo tutto in RGB per avere delle immagini consistenti
        #carichiamo l'immagine utilizzando PIL
        im = Image.open(join(self.data_root,im_path)).convert('RGB')
        
        #se la trasfromazione è definita, applichiamola all'immagine
        if self.transform is not None:
            im = self.transform(im)
            
        #convertiamo l'etichetta in un intero
        label = int(im_label)
        
        #restituiamo un dizionario contenente immagine etichetta
        return {'image' : im, 'label':label}
    
    def mostra(self,i):
        im_path, im_label = self.data.iloc[i]['path'], self.data.iloc[i].label
        #il dataset contiene alcune immagini in scala di grigi
        #convertiamo tutto in RGB per avere delle immagini consistenti
        #carichiamo l'immagine utilizzando PIL
        im = Image.open(join(self.data_root,im_path)).convert('RGB')
        im.show()
    
    def norm(im):
        im = im-im.min()
        return im/im.max()    
    

        