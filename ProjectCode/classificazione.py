# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:44:01 2020

@author: rosaz
"""
import torch
from torch.utils.data import DataLoader
from DatasetClassi import DatasetClassi
from prova import controlFileCSV,controlFileCSVBase, writeJsonModelClass,addValueJsonModel,writeJsonModelInit1,createFolder
from train_class import train_class
from test_classificazione import test_classifier
from test_classifierPair import test_classifierPair
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import numpy as np
from DataSetPairCreate import DataSetPairCreate

def classificazione(directory,filename, version,exp_name,name, model,lr, epochs,  momentum, batch_size, resize):
    print("Classificazione")
    #directory "Class"
    directory =directory
    version=version
    lr=lr
    epochs=epochs
    momentum=momentum
    batch_size = batch_size
    resize=resize
    controlFileCSVBase()
    
    dataSetClass = DatasetClassi(resize)
    dataSetClass.controlNormalize()
    
    train = dataSetClass.dataset_train_norm
    validation = dataSetClass.dataset_valid_norm
    test = dataSetClass.dataset_test_norm
    print("Numeri campioni",len(train))
    createFolder(directory)
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version) 
    
    money_train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True)
    money_test_loader = DataLoader(test, batch_size=batch_size, num_workers=0)
    money_val_loader = DataLoader(validation , batch_size = batch_size, num_workers=0)
    print("Numero di batch", len(money_train_loader))
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_class(directory,version, model, money_train_loader, money_val_loader,resize, batch_size, exp_name , lr=lr, epochs = epochs)
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "batchSize":batch_size }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, contrastiveLoss, accuracy ,time)
    
    path = exp_name+".pth"
    model_test = torch.load(path)
    
    """
    timeTest ,pred , label = test_classifier(model_test,money_test_loader)
    
    accuracyTest= accuracy_score(label, pred)
    print ("Accuracy : %0.2f%%" % (accuracy_score(label, pred)*100))
    precisionTest = precision_score(pred, label)
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pred, label)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pred, label, average=None)
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)
        
    key=["accuracy","precision","recall","mf1_score","time"]
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","timeTest"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,timeTest]
    addValueJsonModel(directory+"modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key[4] ,entry[4], value[4])
   """

def testing_classificazione(directory,path, version,resize,batch_size):
    # directory "Classe
    
    model = torch.load(path)
    controlFileCSV()
    
    dataSetClass = DatasetClassi(resize)
    dataSetClass.controlNormalize()
    
    
    test = dataSetClass.dataset_test_norm
    
    createFolder(directory)
    createFolder(directory+"\\"+version)
    
    money_test_loader = DataLoader(test, batch_size=batch_size, num_workers=0)
    timeTest,pair_prediction, pair_label  = test_classifier(model, money_test_loader)
    
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction,average='micro')
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction,average='micro')
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average='micro')
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)

    key=["accuracy","precision","recall","mf1_score","time"]
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","testing"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,timeTest]
    addValueJsonModel(directory+"\\modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[4] ,entry[4], value[4])
    
    print("Classification Report")
    print(classification_report(pair_label, pair_prediction))
    
    cm = confusion_matrix(pair_label, pair_prediction)
    print("Matrice di confusione \n",cm)
    print("\n")
    #"--------
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)


    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    print("\n")
    print("TNR:",TNR)
    print("FPR:",FPR)
    print("FNR:",FNR)
    print("TPR:",TPR)
    
    #----------------
    

    
    cm.sum(1).reshape(-1,1)
        
    cm=cm/cm.sum(1).reshape(-1,1) #il reshape serve a trasformare il vettore in un vettore colonna
    print("\n")
    print("Matrice di confusione normalizzata \n",cm)
    """
    tnr, fpr, fnr, tpr = cm.ravel()
    print("\n")
    print("TNR:",tnr)
    print("FPR:",fpr)
    print("FNR:",fnr)
    print("TPR:",tpr)
    """
    key = "performance_test"
    entry=["TNR","FPR","FNR","TPR"]
    value=[list(TNR), list(FPR), list(FNR), list(TPR)]
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[0], value[0])
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[1], value[1])
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[2], value[2])
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[3], value[3])



def testing_classificazionePair(directory,path, version,resize,batch_size):
    # directory "Classe
    
    model = torch.load(path)
    controlFileCSV()
    
    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_test = dataSetPair.pair_money_test
    pair_money_test_loader = DataLoader(pair_test, batch_size, num_workers=0)
    
    createFolder(directory)
    createFolder(directory+"\\"+version)
    
   
    timeTest,pair_prediction, pair_label  = test_classifierPair(model, pair_money_test_loader)
    
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction,average='micro')
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction,average='micro')
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average='micro')
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)

    key=["accuracy","precision","recall","mf1_score","time"]
    entry=["accuracyTest_Pair","precisionTest_Pair","recallTest_Pair","f1_score_Test_Pair","testing_Pair"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,timeTest]
    addValueJsonModel(directory+"\\modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[4] ,entry[4], value[4])
    
    print("Classification Report")
    print(classification_report(pair_label, pair_prediction))
    
    cm = confusion_matrix(pair_label, pair_prediction)
    print("Matrice di confusione \n",cm)
    print("\n")
    #"--------
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)


    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    print("\n")
    print("TNR:",TNR)
    print("FPR:",FPR)
    print("FNR:",FNR)
    print("TPR:",TPR)
    
    #----------------
    

    
    cm.sum(1).reshape(-1,1)
        
    cm=cm/cm.sum(1).reshape(-1,1) #il reshape serve a trasformare il vettore in un vettore colonna
    print("\n")
    print("Matrice di confusione normalizzata \n",cm)
    """
    tnr, fpr, fnr, tpr = cm.ravel()
    print("\n")
    print("TNR:",tnr)
    print("FPR:",fpr)
    print("FNR:",fnr)
    print("TPR:",tpr)
    """
    key = "performance_test_Pair"
    entry=["TNR","FPR","FNR","TPR"]
    value=[list(TNR), list(FPR), list(FNR), list(TPR)]
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[0], value[0])
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[1], value[1])
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[2], value[2])
    addValueJsonModel(directory+"\\modelTrained.json",version, key ,entry[3], value[3])








def continue_classificazione(directory,model,version,exp_name,name,lr, momentum,resize,batch_size, epoche_avanza):
    createFolder(directory)
    createFolder(directory+"\\"+version)
    controlFileCSVBase()
    
    dataSetClass = DatasetClassi(resize)
    dataSetClass.controlNormalize()
    
    train = dataSetClass.dataset_train_norm
    validation = dataSetClass.dataset_valid_norm
    test = dataSetClass.dataset_test_norm
    
    
    money_train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True)
    money_test_loader = DataLoader(test, batch_size=batch_size, num_workers=0)
    money_val_loader = DataLoader(validation , batch_size = batch_size, num_workers=0)

    epoche_fatte =0
    array_loss_train=[]
    array_loss_valid=[]
    array_accuracy_train=[]
    array_accuracy_valid=[]
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

    print("Indice epoca gia fatta: ",epoche_fatte)
    print("Epoche avanza :",epoche_avanza)
    print("Array loss train ", array_loss_train)
    print("Array loss valid", array_loss_valid)
    
    print("Array accuracy train", array_accuracy_train)
    print("Array accuracy valid", array_accuracy_valid)
    
    print("Array glb train", array_glb_train)
    print("array glb valid ", array_glb_valid)
    
    epochs = epoche_fatte + epoche_avanza + 1 
    dizionario_array = {"epoche_fatte":epoche_fatte, "epoche_avanza":epoche_avanza,"l_train": array_loss_train, "l_valid": array_loss_valid, "a_train":array_accuracy_train,"a_valid":array_accuracy_valid, "g_train":array_glb_train, "g_valid":array_glb_valid, "tempoTrain":tempoTrain }
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val=train_class(directory,version, model, money_train_loader, money_val_loader,resize, batch_size, exp_name , lr=lr, epochs = epochs, dizionario = dizionario_array)
    
    print("Time computing", f)
    tempo1 = float(tempoTrain)
    tempo2 = float(f) 
    tempo = tempo1+tempo2
    f =str(tempo) 
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "batchSize":batch_size }
    loss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, loss, accuracy ,time)
    
         