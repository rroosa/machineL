# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:20:02 2020

@author: rosaz
"""

from glob import glob 
import os.path
from os.path import basename
from utils.constants import workdir_dataset, image_path, seed, workdir
import os
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import torch
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from CSVImageDataset import CSVImageDataset
from torchvision.transforms import ToTensor
from prova import addKeyValue


class DataSetCreate():
    
    def __init__(self):
        self.path = glob(workdir_dataset) # elenco dei percorsi delle cartelle riferite alle immagini 
                                        #estrarre i nomi delle cartelle dal path completo
        self.classes = [basename(c) for c in self.path]
                                        #dizionario che mappa i nomi delle classi su id numerici (da 0 a 4)
        self.dictionary_name_classes={c: i for i, c in enumerate(self.classes)}
                                        # elenco delle immagini da tutte le cartelle
        self.all_images = [f for f in glob(image_path + "**/*.jpg", recursive=True)]
        #print("List of All image", len(self.all_images))
        self.list_image = self.ListOfAllImages() # lista delle immagini con path images/mome_cartella/img.jpg
        
        self.list_labels = [self.class_from_path(im) for im in self.list_image]
        #print(self.list_labels[:10])
        
        self.dataset_train=0
        self.dataset_test =0
        self.dataset_valid=0
        
        self.run()     
    
    def class_from_path(self,path):
        _, cl, _ = path.split('/')
        return self.dictionary_name_classes[cl]
    
    def name_classes_id(self):
        print(" name_classes : id")
        print(self.dictionary_name_classes)
        return self.dictionary_name_classes
        
    def list_all_images(self):
        print("List path of all images")
        for f in self.all_images:
            print(f)
        return self.all_images
    
    def ListOfAllImages(self):
        list_image = []
        for f in self.all_images:
            f = "/".join(f.split('\\')[1:])
            list_image.append(f)
        return list_image
        #print(self.list_image[:10])
    
    def num_total_file(self):
        self.total_file = len(self.all_images)
        print("Total files = ",self.total_file)
        return self.total_file
    
    def info_classes(self):
        dict_info = [] # lista di oggetti
        size_image = 0
        verify = 0
        h = 0
        w = 0
        c = 0
        for index, path in enumerate (self.path):
            #print(path)
            num_file = len(os.listdir(path))
            #print(num_file)
            
            """Returns the directory size in bytes."""
            total = 0
            
            
            try:
                # ottieni la dimensione della directory
                for  entry in  os.scandir(path):
                    if entry.is_file():
                        if verify ==0:
                            c, h ,w = self.get_size_image(path,entry)
                            verify = 1
                        # se è un file, usa stat() 
                        total += entry.stat().st_size
                        
            except NotADirectoryError:
                # se non è una directory, ottieni  file size then
                return os.path.getsize(path)
            except PermissionError:
                #  errori di apertura della cartella , return 0
                return 0
            total = self.get_size_format(total)
            #print("Total", total)
            #print(self.classes[index])
            #print(self.dictionary_name_classes[self.classes[index]])
            size_image = (h,w,c)
            obj = {"id_class": self.dictionary_name_classes[self.classes[index]],"name_class":self.classes[index],"shape_of_image":size_image,"num_images":num_file,"dim_class":total }
            dict_info.append(obj)
        entry={"dataset":dict_info}
        key = "dataset"
        value=dict_info
        path="Dataset\dataSetJson.json"
        addKeyValue(path,key,value)
        return dict_info
            
    
    def get_size_format(self,b, factor=1024, suffix="B"):
        """
        Scale bytes to its proper byte format
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
        """
        for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
            if b < factor:
                return f"{b:.2f}{unit}{suffix}"
            b /= factor
        return f"{b:.2f}Y{suffix}"
        
    
    def get_size_image(self,path,entry):
        
        image = Image.open(path+'\\'+entry.name)
        print(path+'\\'+entry.name)
        image = ToTensor()(image).unsqueeze(0) # unsqueeze to add artificial first dimension
        #image = Variable(image)
        #image.show()
        #t1 = transforms.ToPILImage()
        #t2 = transforms.ToTensor(image)
        print(image.shape)
        return image.shape[1],image.shape[2],image.shape[3]
    
        
        
        #h,w, c = image.size
        #print(h,w,c)
        #return h,w,c
        
    def run(self):
        #print("1) Dataset of base - DataFrame with columns (path, label)\n")
        self.dataFrame = self.create_DataFrame()
        #print(self.dataFrame.head())
        #print("\n")
        #print("2) Length of DataFrame", self.lengthDataFrame())
        #print("\n")
        
    def create_Dataset_Large(self):
        print("1) Create list Path and list Label")
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        
        #listaIndici=np.random.randint(self.lengthDataFrame(),size=self.lengthDataFrame())
        #print("Lista Indice ", len(listaIndici))
        
        listaIndici=np.random.randint(0,self.lengthDataFrame(),self.lengthDataFrame()*10)
        print("Lista Indice ", len(listaIndici))
        self.listaPercorsi = [self.list_image[i] for i in listaIndici]
        
        self.listaLabel = [self.list_labels[i] for i in listaIndici]
        
        print("2) Create DataFrame Large")
        self.dataset2 = pd.DataFrame({'path':self.listaPercorsi, 'label':self.listaLabel})
        print("2- datset2", len(self.dataset2))
        self.datasetLarge = self.dataFrame.append(self.dataset2,ignore_index=True)
        """QUIIIII"""
        #self.datasetLarge = self.dataFrame
        print("2-datasetLarge",len(self.datasetLarge))
        print("Lengh Dataset Large = ", self.lengthDataFrameLarge())
        print("\n")
        print(self.datasetLarge.head())
        print("\n")
        print("3) Create file 'datasetAll.csv' of Dataset Large ")
        self.datasetLarge.to_csv('Dataset/datasetAll.csv', index=True)
        
        print("4) Split Dataset in TRAIN TEST E VALIDATION")
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.train, self.val, self.test = self.split_train_val_test(self.datasetLarge)
        
                
        if os.path.isfile(workdir+'train.csv') and (os.path.isfile(workdir+'valid.csv')) and (os.path.isfile(workdir+'test.csv')):
            print ("\t Files csv exists, overwrite files")
        
        print("5) Create file csv -  train, val, test ")
        self.train.to_csv(workdir+'train.csv', index=True)
        self.val.to_csv(workdir+'valid.csv', index=True)
        self.test.to_csv(workdir+'test.csv', index=True)
        print("6) Create file csv - id,classes")
        classes, ids = zip(*self.dictionary_name_classes.items())
        classes = pd.DataFrame({'id':ids, 'class':classes}).set_index('id')
        classes.to_csv(workdir+'classes.csv')
            
        print("7) CSV Image Dataset")
        classes = pd.read_csv(workdir+'classes.csv').to_dict()['class']
        self.dataset_train = CSVImageDataset(workdir, workdir+'train.csv')
        self.dataset_valid = CSVImageDataset(workdir, workdir+'valid.csv')
        self.dataset_test = CSVImageDataset(workdir, workdir+'test.csv')
        
        print("8) Example show sample\n")
        sample = self.dataset_train[20]
        print('Class id:',sample['label'], 'Class name:',classes[sample['label']])

        print(sample['image'])
        print("\n")
        self.dataset_train.mostra(20)
       
        # salva nel file Dataset\dataSetJson.json le informazioni del dataset e datasetLarge creati
        listInfo = self.info_classes()
        listInfoLarge = self.info_datasetLarge()
        
        # stampa le informazioni 
        print("\n9) Information dataset Base\n")
        for el in listInfo:
            print(el)
        print("\n10) Information dataset Large\n")
        for el in listInfoLarge:
            print (el)
        
               
    
    def info_datasetLarge(self):
        sumTrain=0
        sumTest=0
        sumVal=0
        
        try:
            with open(workdir+'train.csv') as f:
                next(f)
                sumTrain = sum(1 for line in f)
        except IOError:
            print("File not exists")
            
        try:
            with open(workdir+'test.csv') as f:
                next(f)
                sumTest = sum(1 for line in f)
        except IOError:
                print("File not exists")
        
        try:
            with open(workdir+'valid.csv') as f:
                next(f)
                sumVal = sum(1 for line in f)
        except IOError:
                print("File not exists")
        
        
        dictLarge_info=[]
        obj = {"split": "train", "num_sample":sumTrain}
        dictLarge_info.append(obj)
        obj = {"split": "validation", "num_sample":sumVal}
        dictLarge_info.append(obj)
        obj = {"split": "test", "num_sample":sumTest}
        dictLarge_info.append(obj)
        obj={"num_total":sumTest+sumTrain+sumVal}
        dictLarge_info.append(obj)
        
        #entry={"datasetLarge":dictLarge_info}
        path="Dataset\dataSetJson.json"
        key="datasetLarge"
        value = dictLarge_info
        addKeyValue(path,key,value)
        return dictLarge_info
            
                
    
    def split_train_val_test(self, datasetLarge, perc=[0.2]):
        trainVal, test = train_test_split(datasetLarge, test_size = perc[0])
        train, val = train_test_split(trainVal, test_size = perc[0])
        return train, val, test
    
    
    def lengthTrain(self):
        return len(self.train)
    
    def lengthVal(self):
        return len(self.val)
    
    def lengthTest(self):
        return len(self.test)
        
    
    def lengthDataFrame(self):
        return len(self.dataFrame)
    
    def lengthDataFrameLarge(self):
        return len(self.datasetLarge)

    def create_DataFrame(self):
        dataset = pd.DataFrame({'path':self.list_image, 'label':self.list_labels})
        dataset.head()
        return dataset
 #--------------------------------
    def create_Dataset_Large(self):
        print("1) Create list Path and list Label")
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        listaIndici=np.random.randint(0,self.lengthDataFrame(),self.lengthDataFrame()*10)
        
        self.listaPercorsi = [self.list_image[i] for i in listaIndici]
        
        self.listaLabel = [self.list_labels[i] for i in listaIndici]
        
        print("2) Create DataFrame Large")
        self.dataset2 = pd.DataFrame({'path':self.listaPercorsi, 'label':self.listaLabel})
        
        self.datasetLarge = self.dataFrame.append(self.dataset2,ignore_index=True)
        print("Lengh Dataset Large = ", self.lengthDataFrameLarge())
        print("\n")
        print(self.datasetLarge.head())
        print("\n")
        print("3) Create file 'datasetAll.csv' of Dataset Large ")
        self.datasetLarge.to_csv('Dataset/datasetAll.csv', index=True)
        
        print("4) Split Dataset in TRAIN TEST E VALIDATION")
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.train, self.val, self.test = self.split_train_val_test(self.datasetLarge)
        
                
        if os.path.isfile(workdir+'train.csv') and (os.path.isfile(workdir+'valid.csv')) and (os.path.isfile(workdir+'test.csv')):
            print ("\t Files csv exists, overwrite files")
        
        print("5) Create file csv -  train, val, test ")
        self.train.to_csv(workdir+'train.csv', index=True)
        self.val.to_csv(workdir+'valid.csv', index=True)
        self.test.to_csv(workdir+'test.csv', index=True)
        print("6) Create file csv - id,classes")
        classes, ids = zip(*self.dictionary_name_classes.items())
        classes = pd.DataFrame({'id':ids, 'class':classes}).set_index('id')
        classes.to_csv(workdir+'classes.csv')
            
        print("7) CSV Image Dataset")
        classes = pd.read_csv(workdir+'classes.csv').to_dict()['class']
        self.dataset_train = CSVImageDataset(workdir, workdir+'train.csv')
        self.dataset_valid = CSVImageDataset(workdir, workdir+'valid.csv')
        self.dataset_test = CSVImageDataset(workdir, workdir+'test.csv')
        
        print("8) Example show sample\n")
        sample = self.dataset_train[20]
        print('Class id:',sample['label'], 'Class name:',classes[sample['label']])

        print(sample['image'])
        print("\n")
        self.dataset_train.mostra(20)
       
        # salva nel file Dataset\dataSetJson.json le informazioni del dataset e datasetLarge creati
        listInfo = self.info_classes()
        listInfoLarge = self.info_datasetLarge()
        
        # stampa le informazioni 
        print("\n9) Information dataset Base\n")
        for el in listInfo:
            print(el)
        print("\n10) Information dataset Large\n")
        for el in listInfoLarge:
            print (el)
        
        # ----------------DATASET DI BASE-------------------------
    def create_Dataset_Base(self):
        print("1) Create list Path and list Label")
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        listaIndici=np.random.randint(0,self.lengthDataFrame(),self.lengthDataFrame())
        
        self.listaPercorsi = [self.list_image[i] for i in listaIndici]
        
        self.listaLabel = [self.list_labels[i] for i in listaIndici]
        
        print("2) Create DataFrame Base")
        self.dataset2 = pd.DataFrame({'path':self.listaPercorsi, 'label':self.listaLabel})
        
        #self.datasetLarge = self.dataFrame.append(self.dataset2,ignore_index=True)
        self.datasetBase = self.dataset2
        print("Lengh Dataset Base = ", self.lengthDataFrameBase())
        print("\n")
        print(self.datasetBase.head())
        print("\n")
        print("3) Create file 'datasetAllBase.csv' of Dataset Base ")
        self.datasetBase.to_csv('Dataset/datasetAllBase.csv', index=True)
        
        print("4) Split Dataset in TRAIN TEST E VALIDATION")
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.train, self.val, self.test = self.split_train_val_test(self.datasetBase)
        
                
        if os.path.isfile(workdir+'train_base.csv') and (os.path.isfile(workdir+'valid_base.csv')) and (os.path.isfile(workdir+'test_base.csv')):
            print ("\t Files csv exists, overwrite files")
        
        print("5) Create file csv -  train_base, val_base, test_base")
        self.train.to_csv(workdir+'train_base.csv', index=True)
        self.val.to_csv(workdir+'valid_base.csv', index=True)
        self.test.to_csv(workdir+'test_base.csv', index=True)
        print("6) Create file csv - id,classes")
        classes, ids = zip(*self.dictionary_name_classes.items())
        classes = pd.DataFrame({'id':ids, 'class':classes}).set_index('id')
        classes.to_csv(workdir+'classes.csv')
            
        print("7) CSV Image Dataset")
        classes = pd.read_csv(workdir+'classes.csv').to_dict()['class']
        self.dataset_train = CSVImageDataset(workdir, workdir+'train_base.csv')
        self.dataset_valid = CSVImageDataset(workdir, workdir+'valid_base.csv')
        self.dataset_test = CSVImageDataset(workdir, workdir+'test_base.csv')
        
        print("8) Example show sample\n")
        sample = self.dataset_train[20]
        print('Class id:',sample['label'], 'Class name:',classes[sample['label']])

        print(sample['image'])
        print("\n")
        self.dataset_train.mostra(20)
       
        # salva nel file Dataset\dataSetJson.json le informazioni del dataset 
        listInfo = self.info_classes()
        listInfoBase = self.info_datasetBase()
        
        # stampa le informazioni 
        print("\n9) Information dataset Base\n")
        for el in listInfo:
            print(el)
        print("\n10) Information dataset \n")
        for el in listInfoBase:
            print (el)              
 
    def info_datasetBase(self):
        sumTrain=0
        sumTest=0
        sumVal=0
        
        try:
            with open(workdir+'train_base.csv') as f:
                next(f)
                sumTrain = sum(1 for line in f)
        except IOError:
            print("File not exists")
            
        try:
            with open(workdir+'test_base.csv') as f:
                next(f)
                sumTest = sum(1 for line in f)
        except IOError:
                print("File not exists")
        
        try:
            with open(workdir+'valid_base.csv') as f:
                next(f)
                sumVal = sum(1 for line in f)
        except IOError:
                print("File not exists")
        
        
        dictBase_info=[]
        obj = {"split": "train", "num_sample":sumTrain}
        dictBase_info.append(obj)
        obj = {"split": "validation", "num_sample":sumVal}
        dictBase_info.append(obj)
        obj = {"split": "test", "num_sample":sumTest}
        dictBase_info.append(obj)
        obj={"num_total":sumTest+sumTrain+sumVal}
        dictBase_info.append(obj)
        
        #entry={"datasetLarge":dictLarge_info}
        path="Dataset\dataSetJson.json"
        key="datasetBase"
        value = dictBase_info
        addKeyValue(path,key,value)
        return dictBase_info
    
    
    def info_datasetLarge(self):
        sumTrain=0
        sumTest=0
        sumVal=0
        
        try:
            with open(workdir+'train.csv') as f:
                next(f)
                sumTrain = sum(1 for line in f)
        except IOError:
            print("File not exists")
            
        try:
            with open(workdir+'test.csv') as f:
                next(f)
                sumTest = sum(1 for line in f)
        except IOError:
                print("File not exists")
        
        try:
            with open(workdir+'valid.csv') as f:
                next(f)
                sumVal = sum(1 for line in f)
        except IOError:
                print("File not exists")
        
        
        dictLarge_info=[]
        obj = {"split": "train", "num_sample":sumTrain}
        dictLarge_info.append(obj)
        obj = {"split": "validation", "num_sample":sumVal}
        dictLarge_info.append(obj)
        obj = {"split": "test", "num_sample":sumTest}
        dictLarge_info.append(obj)
        obj={"num_total":sumTest+sumTrain+sumVal}
        dictLarge_info.append(obj)
        
        #entry={"datasetLarge":dictLarge_info}
        path="Dataset\dataSetJson.json"
        key="datasetLarge"
        value = dictLarge_info
        addKeyValue(path,key,value)
        return dictLarge_info
            
                
    
    def split_train_val_test(self, datasetLarge, perc=[0.2]):
        trainVal, test = train_test_split(datasetLarge, test_size = perc[0])
        train, val = train_test_split(trainVal, test_size = perc[0])
        return train, val, test
    
    
    def lengthTrain(self):
        return len(self.train)
    
    def lengthVal(self):
        return len(self.val)
    
    def lengthTest(self):
        return len(self.test)
        
    
    def lengthDataFrame(self):
        return len(self.dataFrame)
    
    def lengthDataFrameLarge(self):
        return len(self.datasetLarge)
    
    def lengthDataFrameBase(self):
        return len(self.datasetBase)
    
    def create_DataFrame(self):
        dataset = pd.DataFrame({'path':self.list_image, 'label':self.list_labels})
        dataset.head()
        return dataset
        
               
        