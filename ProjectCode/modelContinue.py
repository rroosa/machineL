# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:31:34 2020

@author: rosaz
"""


import sys
import errno
from DataSetCreate import DataSetCreate
from utils.constants import workdir 
from sklearn.metrics import accuracy_score
from DataSetPairCreate import DataSetPairCreate
from EmbeddingNet import EmbeddingNet
from torch.utils.data import DataLoader
from utils.constants import workdir 
from train_continue import train_continue
from prova import addValueJsonModel,writeJsonModelClass,net_save, writeJsonModelEpoca, saveArray , writeJsonModel,readJson, plotLoss, writeJsonAccuracy, creteFileJson, addKeyValue, readFileDataset, lengthDataset,controlFileCSVPair, controlFileCSV, createFolder, controlFolder,plotAccuracy,calculateScore
import torch
import json
import numpy as np
from test_siamese_class import test_siamese_class
from test_classifier import test_classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from train_siamese_class import train_siamese_class
from train_siamese_margin_double import train_siamese_margin_double
from train_siamese_margine import train_siamese_margine

def continue_siamese(directory,namefile,version,path,  exp_name,name,model, lr, epochs, momentum, batch_size,resize, margin, logs):
    
    siamese_reload, f, array_loss_train, array_loss_valid, array_sample_train, array_sample_valid,  array_acc_train, array_acc_valid,labels_train,prediction_train,labels_val,prediction_val=train_continue(directory,version,path,exp_name,name, model,lr, epochs, momentum,batch_size,resize, margin, logs)
    
    
    print("time Training\n ", f)
    print("Loss on train", array_loss_train[-1])
    print("Loss on valid\n", array_loss_valid[-1])
    directory= "Model-1-Continue"
    #controlla se è presente la directory, altrimenti la crei
    createFolder(directory)
    #plot 
    plotLoss(directory,namefile,array_loss_train, array_loss_valid, array_sample_train, array_sample_valid)
    plotAccuracy(directory,namefile,array_acc_train,array_acc_valid,array_sample_train,array_sample_valid)
                    
    scoresTrain= calculateScore(labels_train,prediction_train)
    scoresValid = calculateScore(labels_val,prediction_val)
                    
    print("Score on data Train...")
    print("Accuarcy di train: %0.4f"% scoresTrain[0])
    print("Precision di train: %0.4f"% scoresTrain[1])
    print("Recall di train: %0.4f"% scoresTrain[2])
    print("mF1 score di train: %0.4f"% scoresTrain[3])
    
    print("Score on dataValid...")
    print("Accuarcy di validation: %0.4f"% scoresValid[0])
    print("Precision di validation: %0.4f"% scoresValid[1])
    print("Recall di validation: %0.4f"% scoresValid[2])
    print("mF1 score di validation: %0.4f"% scoresValid[3])
    
    valueTime = readJson(directory+"/ModelTrained.json",version,"time","training")
                  
    epoche = readJson(directory+"/ModelTrained.json",version,"hyperparametr","indexEpoch")
                    
    if not valueTime is None:
        f = float(valueTime) + float(f)
                    
    if not epoche is None:
        epochs = epochs + epoche
                        
    hyperparametr = {"indexEpoch":epochs,"lr":lr, "momentum" : momentum, "numSampleTrain": len(labels_train) }
    contrastiveLoss = {"lossTrain": array_loss_train[-1], "lossValid": array_loss_valid[-1] }
    accuracy = {"accuracyTrain":scoresTrain[0] , "accuracyValid":scoresValid[0] }
    precision = {"precisionTrain":scoresTrain[1]  , "precisionValid":scoresValid[1] }
    recall = {"recallTrain":scoresTrain[2] , "recallValid":scoresValid[2] }
    f1score = {"f1_score_Train":scoresTrain[3] , "f1_score_Valid":scoresValid[3]}
    time = {"training": str(f)}
    
    
    writeJsonModel(directory,name,version, hyperparametr, batch_size, contrastiveLoss, accuracy ,f1score, precision, recall, time)   
                        
    
def continue_model_class( directory,filename,version, exp_name,name,model,path_dict, lr,epoche_avanza ,momentum,batch_size,resize,decay=None,margin=1.5,soglia=0.8, modeLoss="single"):
    #exp_name='modello6_v3'
    #path_dict= 'modello6_v3_dict.pth'
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    epoche_fatte =0
    array_loss_train=[]
    array_loss_valid=[]
    array_acc_train=[]
    array_acc_valid=[]
    array_glb_train=[]
    array_glb_valid=[]
        
    percorso1 = directory+"//"+"modelTrained.json"
    with open(percorso1,"r") as file:
        data = json.load(file)
        
    if not( data.get(version) is None):
        obj = data[version]
        if not(obj.get("hyperparametr") is None):
            para = obj["hyperparametr"]
            
            if not(para.get("indexEpoch") is None):
                epoche_fatte = para["indexEpoch"]
    
    percorso2=  directory+"//"+version+"//"+"value_arrays.json"
    with open(percorso2,"r") as file2:
        data2 = json.load(file2)
        
    if not( data2.get("array_loss_train") is None):
        array_loss_train = data2["array_loss_train"]
    
    if not( data2.get("array_loss_valid") is None):
        array_loss_valid = data2["array_loss_valid"]  
    
    if not( data2.get("array_accuracy_train") is None):
        array_accuracy_train = data2["array_accuracy_train"]
        
    if not( data2.get("array_accuracy_valid") is None):
        array_accuracy_valid = data2["array_accuracy_valid"]
    
    
    if not( data2.get("array_glb_train") is None):
        array_glb_train = data2["array_glb_train"]
    
        
    if not( data2.get("array_glb_valid") is None):
        array_glb_valid = data2["array_glb_valid"]
    
    
    
                
        
            
    print("Indice epoca gia fatta: ",epoche_fatte)
    print("Epoche avanza :",epoche_avanza)
    print("Array loss train ", array_loss_train)
    print("Array loss valid", array_loss_valid)
    
    print("Array accuracy train", array_accuracy_train)
    print("Array accuracy valid", array_accuracy_valid)
    
    print("Array glb train", array_glb_train)
    print("array glb valid ", array_glb_valid)
     #indice epoca_fatte 19 
    epochs = epoche_fatte + epoche_avanza + 1 
    dizionario_array = {"epoche_fatte":epoche_fatte, "epoche_avanza":epoche_avanza,"l_train": array_loss_train, "l_valid": array_loss_valid, "a_train":array_accuracy_train,"a_valid":array_accuracy_valid, "g_train":array_glb_train, "g_valid":array_glb_valid}
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val= train_siamese_margine(directory,version,model, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name=exp_name,lr=lr, epochs=epochs, momentum=momentum, margin=margin,soglia=soglia, logdir="logs", decay=decay, modeLoss=modeLoss, dizionario= dizionario_array)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    if not decay is None:
         
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr,"decay":decay, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    else:  
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
    
    """
    checkpoint = torch.load(path_dict)
    model = torch.load(path)
    
    
    #epoche_fatte = checkpoint['epoch']
    print("Epoche fatte",epoche_fatte)
    optimizer = checkpoint['optimizer_state_dict']
    global_step_train = checkpoint['global_step_train']
    global_step_valid = checkpoint['global_step_valid']
    
    percorso = directory+"//"+version+"//"+"value_arrays.json"
    
    array_loss_train = []
    array_loss_valid =[]
    array_accuracy_train = []
    array_accuracy_valid = []
    array_glb_train = []
    array_glb_valid = []
    
    
    with open(percorso,"r") as file:
        data = json.load(file)
        
    if not( data.get("array_loss_train") is None):
        array_loss_train = data["array_loss_train"]
        
    if not(data.get("array_loss_valid") is None):
        array_loss_valid = data["array_loss_valid"]
        
    if not(data.get("array_accuracy_train") is None):
        array_accuracy_train = data["array_accuracy_train"]
        
    if not(data.get("array_accuracy_valid") is None):
        array_accuracy_valid = data["array_accuracy_valid"]
        
    if not(data.get("array_glb_train") is None):
        array_glb_train = data["array_glb_train"]
    
    if not(data.get("array_glb_valid") is None):
        array_glb_valid = data["array_glb_valid"]
    
    dizionario_array = {"epoche_fatte":epoche_fatte, "epoche_avanza":epoche_avanza, "optimizer": optimizer,"l_train": array_loss_train, "l_valid": array_loss_valid, "a_train":array_accuracy_train,"a_valid":array_accuracy_valid, "g_train":array_glb_train, "g_valid":array_glb_valid}
    
    
    print("Training...")
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_class(directory,version, model, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name , lr=lr, epochs = epoche_avanza ,momentum = momentum, dizionario_array = dizionario_array)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    epochs = epoche_avanza + epoche_fatte +1
    
    hyperparametr = {"epochs":epochs,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
  
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, contrastiveLoss, accuracy ,time)
    
    
    siamese_model = torch.load(path)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_class(siamese_model, pair_money_val_loader, margine = None)
    
    numSimilPredette = np.sum(pair_prediction_val==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction_val == 1)
    print("Num Dissimil predette",numDissimilPredette)
    numSimilReali = np.sum(pair_label_val == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label_val==1)
    print("Num Dissimil Reali",numDissimilReali)
    
        #calculate Accuracy
    print(pair_prediction_val[0:10])
    print(pair_label_val[0:10])
    accuracyVal = accuracy_score(pair_label_val, pair_prediction_val)
    print("Accuarcy di test: %0.4f"% accuracyVal)
        #calculate Precision
    precisionVal = precision_score(pair_label_val, pair_prediction_val)
    print("Precision di test: %0.4f"% precisionVal)
        #calculate Recall
    recallVal = recall_score(pair_label_val, pair_prediction_val)
    print("Recall di test: %0.4f"% recallVal)
        #calculate F1 score
    if recallVal!= 0.0 and precisionVal != 0.0:
        
        scores_testing_val = f1_score(pair_label_val,pair_prediction_val, average=None)
        scores_testing_val = scores_testing_val.mean()
        print("mF1 score di testing: %0.4f"% scores_testing_val)
        
        
    else:
        scores_testing_val = 0.000
        print("mscoref1",scores_testing_val)
        
    
    key=["accuracy","precision","recall","mf1_score","time"]
    entry=["accuracyVal","precisionVal","recallVal","f1_score_Val","testVal"]
    value=[accuracyVal,precisionVal,recallVal,scores_testing_val,timeVal]
    addValueJsonModel(directory+"modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key[4] ,entry[4], value[4])
       
    """

def continue_model_margine_double( directory,filename,version, exp_name,name,model, lr,epoche_avanza ,momentum,batch_size,resize,decay=None,margin1=0.8,margin2=1.2, modeLoss="double"):
    
    createFolder(directory+"\\"+version)
    createFolder(directory+"\\"+version+"\\"+"Metod2")
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    epoche_fatte =0
    array_loss_train=[]
    array_loss_valid=[]
    array_acc_train=[]
    array_acc_valid=[]
    array_glb_train=[]
    array_glb_valid=[]
        
    percorso1 = directory+"//"+"modelTrained.json"
    with open(percorso1,"r") as file:
        data = json.load(file)
        
    if not( data.get(version) is None):
        obj = data[version]
        if not(obj.get("hyperparametr") is None):
            para = obj["hyperparametr"]
            
            if not(para.get("indexEpoch") is None):
                epoche_fatte = para["indexEpoch"]
                
        if not(obj.get("time") is None):
            tempo = obj["time"]
            
            if not(tempo.get("training") is None):
                tempoTrain = tempo["training"]
            
            
            
    
    percorso2=  directory+"//"+version+"//"+"value_arrays.json"
    with open(percorso2,"r") as file2:
        data2 = json.load(file2)
        
    if not( data2.get("array_loss_train") is None):
        array_loss_train = data2["array_loss_train"]
    
    if not( data2.get("array_loss_valid") is None):
        array_loss_valid = data2["array_loss_valid"]  
    
    if not( data2.get("array_accuracy_train") is None):
        array_accuracy_train = data2["array_accuracy_train"]
        
    if not( data2.get("array_accuracy_valid") is None):
        array_accuracy_valid = data2["array_accuracy_valid"]
    
    
    if not( data2.get("array_glb_train") is None):
        array_glb_train = data2["array_glb_train"]
    
        
    if not( data2.get("array_glb_valid") is None):
        array_glb_valid = data2["array_glb_valid"]
    
    percorso3=  directory+"//"+version+"//"+"Metod2"+"//"+"value_performance.json"
    with open(percorso3,"r") as file3:
        data3 = json.load(file3)
        
    if not( data3.get("array_acc_valid") is None):
        array_acc_valid_2 = data3["array_acc_valid"]
        
    if not( data3.get("array_f1_valid") is None):
        array_f1_valid_2 = data3["array_f1_valid"]
        
    if not( data3.get("array_recall_valid") is None):
        array_recall_valid_2 = data3["array_recall_valid"]
        
    if not( data3.get("array_precision_valid") is None):
        array_precision_valid_2 = data3["array_precision_valid"]
    
    if not( data3.get("array_tp_valid") is None):
        array_tp_valid_2 = data3["array_tp_valid"]
    
    if not( data3.get("array_fp_valid") is None):
        array_fp_valid_2 = data3["array_fp_valid"]
    
    if not( data3.get("gl_step_valid") is None):
        array_glb_valid_2 = data3["gl_step_valid"]
    
    
        
    
        
    
                
        
            
    print("Indice epoca gia fatta: ",epoche_fatte)
    print("Epoche avanza :",epoche_avanza)
    print("Array loss train ", array_loss_train)
    print("Array loss valid", array_loss_valid)
    
    print("Array accuracy train", array_accuracy_train)
    print("Array accuracy valid", array_accuracy_valid)
    
    print("Array glb train", array_glb_train)
    print("array glb valid ", array_glb_valid)
     #indice epoca_fatte 19 
    epochs = epoche_fatte + epoche_avanza + 1 
    dizionario_array = {"epoche_fatte":epoche_fatte, "epoche_avanza":epoche_avanza,"l_train": array_loss_train, "l_valid": array_loss_valid, "a_train":array_accuracy_train,"a_valid":array_accuracy_valid, "g_train":array_glb_train, "g_valid":array_glb_valid, "tempoTrain":tempoTrain, "array_acc_valid_2":array_acc_valid_2, "array_f1_valid_2":array_f1_valid_2, "array_recall_valid_2":array_recall_valid_2, "array_precision_valid_2":array_precision_valid_2, "array_tp_valid_2":array_tp_valid_2, "array_fp_valid_2":array_fp_valid_2, "array_glb_valid_2":array_glb_valid_2 }
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val= train_siamese_margin_double(directory,version,model, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name=exp_name,lr=lr, epochs=epochs, momentum=momentum, margin1=margin1,margin2=margin2, logdir="logs", decay=decay, modeLoss=modeLoss, dizionario= dizionario_array)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    if not decay is None:
         
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr,"decay":decay, "momentum" : momentum, "numSampleTrain": len(pair_train), "margin1":margin1, "margin2":margin2 }
    else:  
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train),"margin1":margin1, "margin2":margin2 }
    
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    
    time = {"training": f}
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
    
    
    


def continue_model_classif( directory,filename,version, exp_name,name,model,path_dict, lr,epoche_avanza ,momentum,batch_size,resize,decay=None):
    #exp_name='modello6_v3'
    #path_dict= 'modello6_v3_dict.pth'
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    epoche_fatte =0
    array_loss_train=[]
    array_loss_valid=[]
    array_acc_train=[]
    array_acc_valid=[]
    array_glb_train=[]
    array_glb_valid=[]
        
    percorso1 = directory+"//"+"modelTrained.json"
    with open(percorso1,"r") as file:
        data = json.load(file)
        
    if not( data.get(version) is None):
        obj = data[version]
        if not(obj.get("hyperparametr") is None):
            para = obj["hyperparametr"]
            
            if not(para.get("indexEpoch") is None):
                epoche_fatte = para["indexEpoch"]
    
    percorso2=  directory+"//"+version+"//"+"value_arrays.json"
    with open(percorso2,"r") as file2:
        data2 = json.load(file2)
        
    if not( data2.get("array_loss_train") is None):
        array_loss_train = data2["array_loss_train"]
    
    if not( data2.get("array_loss_valid") is None):
        array_loss_valid = data2["array_loss_valid"]  
    
    if not( data2.get("array_accuracy_train") is None):
        array_accuracy_train = data2["array_accuracy_train"]
        
    if not( data2.get("array_accuracy_valid") is None):
        array_accuracy_valid = data2["array_accuracy_valid"]
    
    
    if not( data2.get("array_glb_train") is None):
        array_glb_train = data2["array_glb_train"]
    
        
    if not( data2.get("array_glb_valid") is None):
        array_glb_valid = data2["array_glb_valid"]
    
    
    
                
        
            
    print("Indice epoca gia fatta: ",epoche_fatte)
    print("Epoche avanza :",epoche_avanza)
    print("Array loss train ", array_loss_train)
    print("Array loss valid", array_loss_valid)
    
    print("Array accuracy train", array_accuracy_train)
    print("Array accuracy valid", array_accuracy_valid)
    
    print("Array glb train", array_glb_train)
    print("array glb valid ", array_glb_valid)
     #indice epoca_fatte 19 
    epochs = epoche_fatte + epoche_avanza + 1 
    dizionario_array = {"epoche_fatte":epoche_fatte, "epoche_avanza":epoche_avanza,"l_train": array_loss_train, "l_valid": array_loss_valid, "a_train":array_accuracy_train,"a_valid":array_accuracy_valid, "g_train":array_glb_train, "g_valid":array_glb_valid}
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val= train_siamese_class(directory,version,model, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name=exp_name,decay=decay,lr=lr, epochs=epochs, momentum=momentum,  logdir="logs", dizionario= dizionario_array)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    if not decay is None:
         
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr,"decay":decay, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    else:  
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
    
def continue_model_margine_single( directory,filename,version, exp_name,name,model, lr,epoche_avanza ,momentum,batch_size,resize,decay=None,margin1=2.0,soglia=0.92, modeLoss="single"):

    createFolder(directory+"\\"+version)
    createFolder(directory+"\\"+version+"\\"+"Metod2")
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    epoche_fatte =0
    array_loss_train=[]
    array_loss_valid=[]
    array_acc_train=[]
    array_acc_valid=[]
    array_glb_train=[]
    array_glb_valid=[]
        
    percorso1 = directory+"//"+"modelTrained.json"
    with open(percorso1,"r") as file:
        data = json.load(file)
        
    if not( data.get(version) is None):
        obj = data[version]
        if not(obj.get("hyperparametr") is None):
            para = obj["hyperparametr"]
            
            if not(para.get("indexEpoch") is None):
                epoche_fatte = para["indexEpoch"]
                
        if not(obj.get("time") is None):
            tempo = obj["time"]
            
            if not(tempo.get("training") is None):
                tempoTrain = tempo["training"]
            
            
            
    
    percorso2=  directory+"//"+version+"//"+"value_arrays.json"
    with open(percorso2,"r") as file2:
        data2 = json.load(file2)
        
    if not( data2.get("array_loss_train") is None):
        array_loss_train = data2["array_loss_train"]
    
    if not( data2.get("array_loss_valid") is None):
        array_loss_valid = data2["array_loss_valid"]  
    
    if not( data2.get("array_accuracy_train") is None):
        array_accuracy_train = data2["array_accuracy_train"]
        
    if not( data2.get("array_accuracy_valid") is None):
        array_accuracy_valid = data2["array_accuracy_valid"]
    
    
    if not( data2.get("array_glb_train") is None):
        array_glb_train = data2["array_glb_train"]
    
        
    if not( data2.get("array_glb_valid") is None):
        array_glb_valid = data2["array_glb_valid"]
        """
    percorso3=  directory+"//"+version+"//"+"Metod2"+"//"+"value_performance.json"
    with open(percorso3,"r") as file3:
        data3 = json.load(file3)
        
    if not( data3.get("array_acc_valid") is None):
        array_acc_valid_2 = data3["array_acc_valid"]
        
    if not( data3.get("array_f1_valid") is None):
        array_f1_valid_2 = data3["array_f1_valid"]
        
    if not( data3.get("array_recall_valid") is None):
        array_recall_valid_2 = data3["array_recall_valid"]
        
    if not( data3.get("array_precision_valid") is None):
        array_precision_valid_2 = data3["array_precision_valid"]
    
    if not( data3.get("array_tp_valid") is None):
        array_tp_valid_2 = data3["array_tp_valid"]
    
    if not( data3.get("array_fp_valid") is None):
        array_fp_valid_2 = data3["array_fp_valid"]
    
    if not( data3.get("gl_step_valid") is None):
        array_glb_valid_2 = data3["gl_step_valid"]
    
    """
        
    
        
    
                
        
            
    print("Indice epoca gia fatta: ",epoche_fatte)
    print("Epoche avanza :",epoche_avanza)
    print("Array loss train ", array_loss_train)
    print("Array loss valid", array_loss_valid)
    
    print("Array accuracy train", array_accuracy_train)
    print("Array accuracy valid", array_accuracy_valid)
    
    print("Array glb train", array_glb_train)
    print("array glb valid ", array_glb_valid)
     #indice epoca_fatte 19 
    epochs = epoche_fatte + epoche_avanza + 1 
    dizionario_array = {"epoche_fatte":epoche_fatte, "epoche_avanza":epoche_avanza,"l_train": array_loss_train, "l_valid": array_loss_valid, "a_train":array_accuracy_train,"a_valid":array_accuracy_valid, "g_train":array_glb_train, "g_valid":array_glb_valid, "tempoTrain":tempoTrain }
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val= train_siamese_margine(directory,version,model, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name=exp_name,lr=lr, epochs=epochs, momentum=momentum, margin=margin1,soglia=soglia, logdir="logs", decay=decay, modeLoss=modeLoss, dizionario= dizionario_array)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    if not decay is None:
         
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr,"decay":decay, "momentum" : momentum, "numSampleTrain": len(pair_train), "soglia":soglia, "soglia":soglia }
    else:  
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train),"margin1":margin1, "soglia":soglia}
    
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    
    time = {"training": f}
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
    
           