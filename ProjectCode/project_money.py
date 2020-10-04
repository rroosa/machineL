# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: rosaz
"""

import sys
import errno
from DataSetCreate import DataSetCreate
from utils.constants import workdir 
from sklearn.metrics import accuracy_score
from DataSetPairCreate import DataSetPairCreate
from EmbeddingNet import EmbeddingNet
from MiniAlexNet import MiniAlexNet
from torchvision.models import resnet34
from SiameseNet import SiameseNetwork
from classificazione import classificazione,testing_classificazione,testing_classificazionePair, continue_classificazione
from MNet import MNet
from Demo import Demo
from ModelM import ModelM
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from train_siamese import train_siamese
from modelContinue import continue_siamese, continue_model_margine_single, continue_model_class, continue_model_margine_double, continue_model_classif
from test_siamese import test_siamese
from prova import writeJsonModel,readJson, plotLoss, writeJsonAccuracy, creteFileJson, addKeyValue, readFileDataset, lengthDataset,controlFileCSVPair, controlFileCSV, createFolder, controlFolder,plotAccuracy,calculateScore
from trainModel import train_model,train_model_class_v1, train_model_margine, train_model_class_adam, train_model_margine_dynamik, distribution, train_model_margine_distrib, train_model_margine_diff, train_model_margine_double
from modelTest import test_model, test_model_class, test_model_margine, test_model_margine_dynamik, test_model_performance,test_model_margine_double
from train_continue import train_continue
from gaussian_distribution import gaussian_distribution_train_margine_single
import argparse
import sys
import torch
from torch import nn
from copy import deepcopy




def main(argv):

    #crazione file "dataSet.json" se non esiste
    entry={"nameDB":"Moneys"}
    controlFolder("Dataset")
    creteFileJson("Dataset\dataSetJson.json",entry)
    data_create = DataSetCreate()

    #name_id = data_create.name_classes_id()
    #list_all_images = data_create.list_all_images()
    #num_tot_files = data_create.num_total_file()
      
    parser = argparse.ArgumentParser( description = "Dataset Money")
    
    parser.add_argument('--create', help="datasetBase | datasetLarge | datasetPair")
    parser.add_argument('--info', help="dataset | datasetBase | datasetLarge | datasetPair")
    #parser.add_argument('--training', help="1")
    parser.add_argument('--test', help="Name of model [model5 | model6 ]", type=str)
    
    parser.add_argument('--train', help = "Name of model [model5 | model6 ]", type=str)
    parser.add_argument('--v', help ="version" , type=int)

    parser.add_argument('--progress', help = "Name of model [model5 | model6 ]", type=str)
    parser.add_argument('--file', help = "name file .pth", type=str)
    
    parser.add_argument('--e', help ="epoche" , type=int)
    parser.add_argument('--margine', help ="dim of resize" , type=int)
    
    parser.add_argument('--classification', help ="[ train | test | continue | demo ]", type=str )
    
    parser.add_argument('--classtest', help ="classTest" )
    parser.add_argument('--demo', help="[ model5 | model6 ]")
    parser.add_argument('--pair',help=" insert id pair [0 - 13824]",type=int)
    parser.add_argument('--soglia',help="soglia",type=float)
    parser.add_argument('--margin', help="margin", type=float)
    
    parser.add_argument('--path', help="path of model '.pth'", type=str)
    
    parser.add_argument('--distribution', help="distribuzione dei dati di train allenati")
    
    
    parser.add_argument('--pathModel', help="percorso modello da inizializzare")
    
    parser.add_argument('--margin1', help="margine 1", type=float)
    
    parser.add_argument('--margin2', help="margine 2", type=float)
    
    parser.add_argument('--roc', help="roc")
    argomento = parser.parse_args()
    
    
#-------- DISTRIBUTION------
     
    required_together_distrib = ('distribution','v')
    if argomento.distribution is not None:
        # args.model will be None if v is not provided
        if not all([getattr(argomento,x) for x in required_together_distrib]):
            raise RuntimeError("Cannot supply --distribution without --v ")
        else:
                            
            #------  MODEL 6 
            if argomento.distribution == "model6":
                print("DISTRIBUTION model ",argomento.distribution)
                #------       MODEL 6 v 2
                if argomento.v == 2:
                    print("version v2")  
                    directory= "Model-6"
                    version="2"
                    resize = 100
                    batch_size= 16
                    createFolder(directory)
                    createFolder(directory+"\\"+version)
                    dataSetPair = DataSetPairCreate(resize)
                    dataSetPair.controlNormalize()
                    pair_train = dataSetPair.pair_money_train
                                
                    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
                    path = directory+"\\"+version+"\\modello6_v2_6.pth"
                    gaussian_distribution_train_margine_single(directory, version, pair_money_train_loader,resize,batch_size, path)
                else:
                    exit(0)


#--------------------------- DEMO -------------------------------
     
    required_together_demo = ('demo','v','pair')
    if argomento.demo is not None:
        # args.model will be None if v is not provided
        if not all([getattr(argomento,x) for x in required_together_demo]):
            raise RuntimeError("Cannot supply --demo without --v --pair")
        else:
            
            #------  MODEL 5 
            if argomento.demo == "model5":
                print("Demo model ",argomento.demo)   
    
    
                if argomento.v == 5:
                    print("version v5")
                    print("model5 v5 ResNet siamese classification SGD")
                    directory = "Model-5\\"
                    path= 'modello5_v5.pth'
                    version="5"
                    idPair = argomento.pair
                    # verifica l'id corrispondente alla coppia se è presente
                    resize = 100
                    demo_obj = Demo(directory, version, resize)
                    demo_obj.controlPair(idPair)
                    demo_obj.read_normalize()
                    dizionario = demo_obj.getitem(idPair)
                    
                    siamese_test = torch.load(path)
                    
                    demo_obj.test_demo(dizionario, siamese_test)
                    demo_obj.plottare()
                
                elif argomento.v == 7:
                    print("version v7")
                    print("DEMO model5 v7, Marek Net siamese classification SGD")
                    directory = "Model-5\\"
                    version="7"
                    path= directory+version+"\\"+'modello5_v7_17.pth'

                    idPair = argomento.pair
                    # verifica l'id corrispondente alla coppia se è presente
                    resize = 100
                    demo_obj = Demo(directory, version, resize)
                    demo_obj.controlPair(idPair)
                    demo_obj.read_normalize()
                    dizionario = demo_obj.getitem(idPair)
                    
                    siamese_test = torch.load(path)
                    
                    demo_obj.test_demo(dizionario, siamese_test)
                    demo_obj.plottare()
                    
                else:
                    print("Versione del model5 non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --train model5 --v [ 5 | 7 ]\n")
                    exit(0)
                    
                    
            # --DEMO ---- MODEL 6 
            elif argomento.demo == "model6":
                print("Demo model ",argomento.demo)
                
                #------DEMO---  MODEL 6 v 2
                if argomento.v == 2:
                    print("version v2")
                    print("model6 v2 ResNet, single margine=2.0, soglia=0.92")
                    directory = "Model-6\\"
                    version="2"
                    path= directory+version+"\\"+'modello6_v2_6.pth'
                    
                    idPair = argomento.pair
                    resize=100
                    demo_obj = Demo(directory, version, resize)
                    demo_obj.controlPair(idPair)
                    demo_obj.read_normalize()
                    dizionario = demo_obj.getitem(idPair)
                    
                    siamese_test = torch.load(path)
                    soglia = 0.92
                    dist = demo_obj.test_demo_single_margine(dizionario, siamese_test,soglia)
                    demo_obj.plottare(dist)
        
                elif argomento.v == 4:
                    print("version v2")
                    print("model6 v4 ResNet, double margine=0.7 e 1.3")
                    directory = "Model-6\\"
                    version="4"
                    path= directory+version+"\\"+'modello6_v4_51.pth'
                    
                    idPair = argomento.pair
                    resize=100
                    demo_obj = Demo(directory, version, resize)
                    demo_obj.controlPair(idPair)
                    demo_obj.read_normalize()
                    dizionario = demo_obj.getitem(idPair)
                    
                    siamese_test = torch.load(path)
                    margin1=0.7
                    margin2=1.2
                    dist = demo_obj.test_demo_double_margine(dizionario, siamese_test,margin1, margin2)
                    demo_obj.plottare(dist)  
                
                else:
                    print("Versione del model6 non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --train model6 --v [ 2 | 4 ]\n")
                    exit(0) 
                
            else:
                print("Modello non riconosciuto")
                sys.stderr.write("Model not acknowledged, try --train [ model5 | model6 ]\n")
                exit(0)                 

# --------------------TRAIN------CLASSIFICAZIONE DEI DATI ---------------------
          
    elif argomento.classification == "train":
        required_together = ('classification','v','e')
        if not all([getattr(argomento,x) for x in required_together]):
            raise RuntimeError("Cannot supply --classification train without --v --e")
        else:
               #-------MODEL MNET
                if argomento.v == 2:
                    epochs=20
                    lr = 0.0001
                    momentum=0.9
                    batch_size=16
                    resize=100
                    if argomento.e is not None:
                        epochs = argomento.e

                    directory="Classe"
                    filename="//class"
                    version="2"
                    exp_name = 'class_2'
                    name='ModelM'
                    model = ModelM()
        
                    classificazione( directory,filename, version,exp_name,name, model,lr, epochs,  momentum, batch_size, resize)
                
                #--------MODEL RESNET
                elif argomento.v == 1:
                    print("Resnet")
                    lr = 0.0001
                    momentum=0.9
                    batch_size=16
                    resize=256
                    if argomento.e is not None:
                        epochs = argomento.e
                    directory="Classe"
                    filename="//class"
                    version="1"
                    exp_name = 'class_1'
                    name='ResNet'
                    
                    model =  resnet34(pretrained=True)
                    resnet_copy = deepcopy(model)
                
                    ### adattamento
                    num_class = 5
                    resnet_copy.fc = nn.Linear(512, num_class)
                    resnet_copy.num_classes = num_class
                    print(resnet_copy)
                    
        
                    classificazione( directory,filename, version,exp_name,name,resnet_copy,lr, epochs,  momentum, batch_size, resize)
                
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --classification train --v [ 1 | 2 ]\n")
                    exit(0)

#-------------------TEST ------CLASSIFICAZIONE DEI DATI 
                
                #---- test su data set di base 
    elif argomento.classification == "test":
        required_together = ('classification','v')
        if not all([getattr(argomento,x) for x in required_together]):
            raise RuntimeError("Cannot supply --classification test without --v")
        else:
                if argomento.v == 2:
                    print("MNet classification version 2")
                    directory="Classe"
                    
                    version="2"
                    batch_size=16
                    resize=100
                    name='ModelM'
                    
                    if argomento.pathModel is not None:
                        path_dict = argomento.pathModel
                    else:
                        path_dict= 'Classe//2//class_2_44.pth'
                        
                    testing_classificazione(directory,path_dict, version,resize,batch_size)
                    
                elif argomento.v == 1:
                    print("Resnet classification version 1")
                    
                    directory="Classe"
                
                    version="1"
                    if argomento.pathModel is not None:
                        path_dict = argomento.pathModel
                    else:
                        path_dict= 'Classe//1//class_1_19.pth'
                    name='ResNet'
                    batch_size =4 
                    resize =256
                    testing_classificazione(directory,path_dict, version,resize,batch_size)
                
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --classification test --v [ 1 | 2 ]\n")
                    exit(0)

#---------------TEST   su datasetPair con classificazione Manuale

    elif argomento.classification == "testPair":
        required_together = ('classification','v')
        if not all([getattr(argomento,x) for x in required_together]):
            raise RuntimeError("Cannot supply --classification testPair without --v")
        else:
                if argomento.v == 2:
                    
                    directory="Classe"
                    
                    version="2"
                    batch_size=16
                    resize=100
                    name='ModelM'
                    
                    if argomento.pathModel is not None:
                        path_dict = argomento.pathModel
                    else:
                        path_dict= 'Classe//2//class_2_44.pth'
                        
                    testing_classificazionePair(directory,path_dict, version,resize,batch_size)
                    
                elif argomento.v == 1:
                    print("Resnet classification version 1")
                    
                    directory="Classe"
                
                    version="1"
                    if argomento.pathModel is not None:
                        path_dict = argomento.pathModel
                    else:
                        path_dict= 'Classe//1//class_1_19.pth'
                    name='ResNet'
                    batch_size =4 
                    resize =256
                    testing_classificazionePair(directory,path_dict, version,resize,batch_size)
                
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --classification testPair --v [ 1 | 2 ]\n")
                    exit(0) 
                       
 #-------------------CONTINUE ------CLASSIFICAZIONE DEI DATI
                    
    elif argomento.classification == "continue":
                required_together = ('classification','v','e')
                if not all([getattr(argomento,x) for x in required_together]):
                    raise RuntimeError("Cannot supply --classification continue without --v --e")
                else:
                    
                    if argomento.v == 2:
                        print("MNet classification continue version 2")
                        directory="Classe"
                        exp_name = 'class_2'
                        version="2"
                        lr = 0.0001
                        momentum=0.9
                        batch_size=16
                        resize=100
                        
                        name='ModelM'
                    
                        if argomento.pathModel is not None:
                            path_dict = argomento.pathModel
                        else:
                            path_dict= 'Classe//2//class_2_19.pth'
                            
                        model = torch.load(path_dict) 
                        epoche_avanza = argomento.e
                        continue_classificazione(directory,model, version,exp_name,name,lr, momentum,resize,batch_size, epoche_avanza)
                    
                    elif argomento.v == 1:
                        print("Resnet classification continue version 1")
                    
                        directory="Classe"
                        version="1"
                        batch_size =4 
                        resize =256
                        lr = 0.0001
                        momentum=0.9
                        exp_name = 'class_1'
                        name='ResNet'
                        if argomento.pathModel is not None:
                            path_dict = argomento.pathModel
                        else:
                            path_dict= 'Classe//1//class_1_19.pth'
                    
                        model = torch.load(path_dict)
                        epoche_avanza = argomento.e
                    
                        continue_classificazione(directory,model,version,exp_name,name,lr,momentum,resize,batch_size, epoche_avanza)
                    
                    else:
                        print("Versione non riconosciuta")
                        sys.stderr.write("Version not acknowledged, try --classification continue --v [ 1 | 2 ]\n")
                        exit(0)          
                                
 # --------------- DEMO ------------------CLASSIFICAZIONE MANUAL
    elif argomento.classification =="demo":
                
                required_together = ('classification','v','pair')
                if not all([getattr(argomento,x) for x in required_together]):
                    raise RuntimeError("Cannot supply --classification demo without --v --pair")
                else:
                    
                     #----- MODEL RESNET  
                    if argomento.v == 1:
                        print("Classification Manual ResNet")
                        if argomento.pathModel is not None:
                            path = argomento.pathModel
                        else:
                            path= 'Classe\\1\\class_1_19.pth'
                        directory = "Classe\\"
                        version="1"
                        idPair = argomento.pair
                        
                        resize = 256
                        demo_obj = Demo(directory, version, resize)
                        demo_obj.controlPair(idPair)
                        demo_obj.read_normalize()
                        dizionario = demo_obj.getitem(idPair)
                    
                        class_test = torch.load(path)
                    
                        demo_obj.test_demo_order_manual(dizionario, class_test)
                        demo_obj.plottare()
                        
                        
                        
                     #----- MODEL MNET   
                    elif argomento.v == 2:
                        directory = "Classe\\"
                        if argomento.pathModel is not None:
                            path = argomento.pathModel
                        else:
                            path= 'Classe\\2\\class_2_44.pth'
                        
                        version="2"
                        idPair = argomento.pair
                        # verifica l'id corrispondente alla coppia se è presente
                        resize = 100
                        demo_obj = Demo(directory, version, resize)
                        demo_obj.controlPair(idPair)
                        demo_obj.read_normalize()
                        dizionario = demo_obj.getitem(idPair)
                    
                        class_test = torch.load(path)
                    
                        demo_obj.test_demo_order_manual(dizionario, class_test)
                        demo_obj.plottare()     
                        return
                    
                    else:
                        print("Versione non riconosciuta")
                        sys.stderr.write("Version not acknowledged, try --classification demo --v [ 1 | 2 ]\n")
                        exit(0)   

                    
#------------ CREAZIONE DEI DATASET --create 
    
    if argomento.create =="datasetBase": # creazione dataset di Base
        data_create.create_Dataset_Base()
    
    elif argomento.create =="datasetLarge": # creazione dataset di Base e datasetLarge
        data_create.create_Dataset_Large()
    
    elif argomento.create == "datasetPair":
        # controlla se è presente il dataset splittato
        controlFileCSV()
        dataSetPair = DataSetPairCreate()
        dataSetPair.controlNormalize()

 #-------------   INFORMAZIONI SUI DATASET  --info
        
    data_set_info = argomento.info
    if(data_set_info == "dataset"):
        #oggetto DataSetCreate
        print("Dataset of base\n")
        #lettura da file Dataset\dataSetJson.json
        info= readFileDataset("Dataset\dataSetJson.json","dataset")
        #info = data_create.info_classes()
        for i in info:
            print(i)
        num = lengthDataset("Dataset\dataSetJson.json","dataset","num_images")
        print("Length Dataset of Base = ",num)
        print("\n")
        
    elif(data_set_info == "datasetBase"):
        print("Dataset Base\n")
        #info = data_create.info_datasetLarge()
        info= readFileDataset("Dataset\dataSetJson.json","datasetBase")
        for i in info:
            print(i)
        num = lengthDataset("Dataset\dataSetJson.json","datasetBase","num_sample")
        print("Length DatasetBase = ",num)      
        
    elif(data_set_info == "datasetLarge"):
        print("Dataset Large\n")
        #info = data_create.info_datasetLarge()
        info= readFileDataset("Dataset\dataSetJson.json","datasetLarge")
        for i in info:
            print(i)
        num = lengthDataset("Dataset\dataSetJson.json","datasetLarge","num_sample")
        print("Length DatasetLarge = ",num)
    
    elif(data_set_info == "datasetPair"):
        
        print("DatasetPair\n")
        info = readFileDataset("Dataset\dataSetJson.json","dataSetPair")
        for i in info:
            print(i)
            
#--------------FASE TRAINING OF MODEL 5 and 6 --train 
    
    required_together = ('train','v','e')
    if argomento.train is not None:
        
        if not all([getattr(argomento,x) for x in required_together]):
            raise RuntimeError("Cannot supply --train without --v --e")
        else:
                            
            #------  MODEL 5 
            if argomento.train == "model5":
                if argomento.v == 7:               
                    # siamese con trasfer-learning Mnet usata pe rla classigìficazione
                    # tolto il livello per la classificazione a 2 classi
                    # nuovo- inizializzazione class_2 epoche 44
                    lr = 0.0001
                    momentum = 0.9
                    resize = 100
                    epochs = 20
                    if argomento.e is not None:
                        epochs = argomento.e
                    batch_size= 4
                    directory="Model-5"
                    filename="//5_v7"
                    version="7"
                    exp_name='modello5_v7'
                    name='MNet'
                    
                    #inizializzazione del modello con parametri di MNet
                    path ="class_2.pth"
                    model = torch.load(path)
                    model_copy = deepcopy(model)
                    fully_connect = model_copy.fc
                    fully = list(fully_connect)
                    fully.pop()
                    model_copy.fc=nn.Sequential(*fully)
                    # adattamento 
                    model_copy.fc2 = nn.Sequential(nn.Linear(512, 2))
                    print(model_copy)
                
                    train_model_class_v1(directory,filename,version,exp_name ,name, model_copy, lr,epochs,momentum,batch_size,resize)

                elif argomento.v == 5:
                    # siamese con trasfer-learning Resnet usato per la classificazione
                    # e tolto l'ultimo livello
                    # aggiunto per prendere in ingresso la concatenazione degli output
                    # e aggiunto il livello per la classificazione a 2 classi
                    # la loss function è la CrossEntropy per la classificazione 0 e 1
                    lr = 0.0001
                    momentum = 0.9
                    resize = 100
                    epochs = 20
                    if argomento.e is not None:
                        epochs = argomento.e
                    batch_size= 4
                    decay=0.0004
                    directory="Model-5"
                    filename="//5_v5"
                    version="5"
                    exp_name='modello5_v5'
                    name='ResNet_Class'
                     
                    # inizializzazione
                    model = torch.load("class_1.pth")
                    model_copy = deepcopy(model)
                   
                    ### adattamento
                    num_class = 256
                    model_copy.fc = nn.Linear(512, num_class)
                    model_copy.num_classes = num_class
                    print(model_copy)
                    
                    model_copy.fc2 = nn.Sequential(nn.Linear(512, 2))
                    print(model_copy)
                    
                    train_model_class_v1(directory,filename,version,exp_name ,name,model_copy, lr,epochs,momentum,batch_size,resize,decay=decay,modeLoss=None, dizionario_array = None)
                
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --train model5 --v [ 5 | 7 ]\n")
                    exit(0)    
                         
                #-----train MODEL6  siamese
            elif argomento.train == "model6":
                #----- Resnet - single margine
                if argomento.v == 2:
                    # siamese con trasfer-learning Resnet usato per la classificazione
                    # e tolto il livello per la classivicazione a 5 classi
                    # e inserito quello da 256
                    # 
                    # la loss function è la Contrastive loss , margine 

                    decay=0.0004                    
                    lr = 0.0001
                    momentum = 0.9
                    resize = 100
                    epochs = 20
                    if argomento.e is not None:
                        epochs = argomento.e
                    batch_size= 4
                    directory="Model-6"
                    filename="//6_v2"
                    version="2"
                    exp_name='modello6_v2'
                    name='RestNet_Margine'
                    
                    # Usato per la classificazione a 5 classi, fine-tuning Resnet34
                    model = torch.load("class_1.pth")
                    model_copy = deepcopy(model)
                                      
                    ### adattamento
                    num_class = 256
                    model_copy.fc = nn.Linear(512, num_class)
                    model_copy.num_classes = num_class
                    print(model_copy)

                    train_model_margine(directory,filename,version,exp_name ,name, model_copy, lr,epochs,momentum,batch_size,resize, decay=decay,margin=2.0,soglia=1.0, modeLoss="single")

                elif argomento.v == 4:
                    # siamese con trasfer-learning Resnet usato per la classificazione
                    # e tolto il livello per la classivicazione a 5 classi
                    # e inserito quello da 256
                    # la loss function è la Contrastive loss , double margine 

                    decay=0.004                    
                    lr = 0.0001
                    momentum = 0.9
                    resize = 100
                    epochs = 20
                    if argomento.e is not None:
                        epochs = argomento.e
                    batch_size= 4
                    directory="Model-6"
                    filename="//6_v4"
                    version="4"
                    exp_name='modello6_v4'
                    name='RestNet_Margine_Double'
                    
                    model = torch.load("class_1.pth") # Usato per la classificazione fine-tuning Resnet
                    model_copy = deepcopy(model)
                    # serve per rimuovere l'ultimo livello"
                    ### adattamento
                    num_class = 256
                    model_copy.fc = nn.Linear(512, num_class)
                    model_copy.num_classes = num_class
                    print(model_copy)
                    margin1 = 0.7
                    margin2 = 1.2
                    if argomento.margin1 is not None:
                        margin1 = argomento.margin1
                                        
                    if argomento.margin2 is not None:
                        margin2 = argomento.margin2
                    
                    train_model_margine_double(directory,filename,version,exp_name ,name, model_copy, lr,epochs,momentum,batch_size,resize, decay=decay,margin1=margin1,margin2=margin2 , modeLoss="double")
                
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --train model6 --v [ 2 | 4 ]\n")
                    exit(0)            
            
            else:
                print("Modello non riconosciuto ")
                sys.stderr.write("Model not acknowledged, try --train [model5 | model6 ]\n")
                exit(0) 

#--------------FASE  TESTING --test 
    required_together_test = ('test','v')
    
    if argomento.test is not None:
        if not all([getattr(argomento,x) for x in required_together_test]):
            raise RuntimeError("Cannot supply --test without --v")
        else:
            #------ test MODEL 5 
            if argomento.test =="model5":
                
                # ----------model 5 v 5 ---- ResNet
                if argomento.v == 5:
                    print("version", argomento.v)
         
                    print("model 5 v5 ResNet classi siamese con lr =0.0001 ")
                    directory = "Model-5\\"
                    path= 'modello5_v5.pth'
                    version="5"
                    
                    batch_size = 16
                    resize = 100
                    
                    test_model_class(directory,path, version,resize,batch_size, margine=None)
                
                elif argomento.v == 7:
                    # ----------model 5 v 7 ---- MNet
                    print("version", argomento.v)
         
                    print("model 5 v7 MNet classi siamese con lr =0.0001")
                    directory = "Model-5\\"
                    
                    path= 'Model-5\\7\\modello5_v7_17.pth'
                    version="7"
                    
                    batch_size = 16
                    resize = 100
                    
                    test_model_class(directory,path, version,resize,batch_size, margine=None)
                    
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --test model5 --v [ 5 | 7 ]\n")
                    exit(0)                
              
                #----------test  MODEL 6
            elif argomento.test == "model6":
                
                    #------ model test 6 v 2
                if argomento.v == 2:
                    print("version", argomento.v)
                    print("model6 v2 Test ResNet siamese margine one 2.0 soglia 0.92")
                    directory = "Model-6\\"
                    path= directory+"2\\"+'modello6_v2_6.pth'
                    version="2"
                    soglia = 0.92
                    if argomento.soglia is not None:
                        soglia = argomento.soglia
                    batch_size = 16
                    resize =100
                    print("Soglia",soglia)
                    test_model_margine(directory,path, version,resize,batch_size, margine=soglia)

                    #-------- model test 6 v 4 
                elif argomento.v == 4:
                    print("version", argomento.v)
                    print("model6 v 4 Test ResNet siamese margine double 0.7 e 1.2, numero epoche 52 ")
                    directory = "Model-6\\"
                    path= directory+"4\\"+'modello6_v4_51.pth'
                    version="4"
                    margin1 = 0.7
                    margin2 = 1.2
                    batch_size = 16
                    resize =100
                    
                    test_model_margine_double(directory,path, version,resize,batch_size, margin1,margin2)
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --test model6 --v [ 2 | 4 ]\n")
                    exit(0)                     
                                            
            else:
                print("Modello non riconosciuto")
                sys.stderr.write("Model not acknowledged, try --test [model5 | model6 ]\n")
                exit(0)                    

# ---------------------PERFORMANCE


    if argomento.roc is not None:
        print(argomento.roc)
        print(argomento.v)

            #  PERFORMANCE MODEL 6 V 2
        if argomento.roc == "model6":
                                # model test 6 v 2
                if argomento.v == 2:
                    print("version", argomento.v)
                    print("model6 v2 Test ResNet siamese margine one 2.0 soglia 1.0")
                    directory = "Model-6"
                    version = "2"
                    path= directory+"\\"+version+'\\modello6_v2_6.pth'
                    version="2"
 
                    batch_size = 16
                    resize =100
                    
                    test_model_performance(directory,path, version,resize,batch_size)
                        
#--------------FASE  CONTINUE --continue 
    required_together_continue = ('progress','v','e')
    # args.model will be None if v is not provided
    if argomento.progress is not None:
        if not all([getattr(argomento,x) for x in required_together_continue]):
            raise RuntimeError("Cannot supply --progress without --v --e")
        else:
            
                #----- MODEL 6
            if argomento.progress == "model6":
                print("model", argomento.progress)
                
                # model continue 6 v 2
                if argomento.v == 2:
                    # siamese con trasfer-learning Resnet usato per la classificazione
                    # e tolto il livello per la classivicazione a 5 classi
                    # e inserito quello da 256
                    # la loss function è la Contrastive loss , margine single
                    
                    decay=0.0004                    
                    lr = 0.0001
                    momentum = 0.9
                    resize = 100
                    
                    batch_size= 4
                    directory="Model-6"
                    filename="//6_v2"
                    version="2"
                    exp_name='modello6_v2'
                    name='RestNet_Margine_Single'
                    if argomento.pathModel is not None:
                        path = argomento.pathModel
                    else:                
                        path = 'Model-6//2//modello6_v2_13.pth'
                    model = torch.load(path)

                    epoche_avanza = argomento.e
                    continue_model_margine_single(directory,filename,version,exp_name ,name, model, lr,epoche_avanza,momentum,batch_size,resize, decay=decay,margin1=2.0,soglia=0.92 , modeLoss="single")
                                      
                # model continue 6 v 4
                elif argomento.v == 4:
                    # siamese con trasfer-learning Resnet usato per la classificazione
                    # e tolto il livello per la classivicazione a 5 classi
                    # e inserito quello da 256
                    # la loss function è la Contrastive loss , margine double

                    decay=0.004                    
                    lr = 0.0001
                    momentum = 0.9
                    resize = 100
                    
                    batch_size= 4
                    directory="Model-6"
                    filename="//6_v4"
                    version="4"
                    exp_name='modello6_v4'
                    name='RestNet_Margine_Double'
                    if argomento.pathModel is not None:
                        path = argomento.pathModel
                    else:
                        path = 'Model-6//4//modello6_v4_56.pth'
                    model = torch.load(path)

                    margin1 = 0.7
                    margin2 = 1.2
                    if argomento.margin1 is not None:
                        margin1 = argomento.margin1
                                        
                    if argomento.margin2 is not None:
                        margin2 = argomento.margin2
                    epoche_avanza = argomento.e
                    continue_model_margine_double(directory,filename,version,exp_name ,name, model, lr,epoche_avanza,momentum,batch_size,resize, decay=decay,margin1=margin1,margin2=margin2 , modeLoss="double")
                
                # model continue 6 v 6
                elif argomento.v == 6:

                    decay=0.02                    
                    lr = 0.001
                    momentum = 0.9
                    resize = 100
                    
                    batch_size= 4
                    directory="Model-6"
                    filename="//6_v6"
                    version="6"
                    exp_name='modello6_v6'
                    name='RestNet_Margine_Double'
                    if argomento.pathModel is not None:
                        path = argomento.pathModel
                    else:
                        path = 'Model-6//4//modello6_v4_51.pth'
                    model = torch.load(path)

                    margin1 = 0.7
                    margin2 = 1.2
                    if argomento.margin1 is not None:
                        margin1 = argomento.margin1
                                        
                    if argomento.margin2 is not None:
                        margin2 = argomento.margin2
                    epoche_avanza = argomento.e
                    continue_model_margine_double(directory,filename,version,exp_name ,name, model, lr,epoche_avanza,momentum,batch_size,resize, decay=decay,margin1=margin1,margin2=margin2 , modeLoss="double")
                else:
                    print("Versione non riconosciuta")
                    sys.stderr.write("Version not acknowledged, try --progress model6 --v [ 2 | 4 | 6 ]\n")
                    exit(0)                       
            else:
                print("Modello non riconosciuto ")
                sys.stderr.write("Model not acknowledged, try --progress [ model6 ]\n")
                exit(0)  

if __name__ == "__main__":
    main(sys.argv[1:])  
