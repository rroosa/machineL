# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:09:06 2020

@author: rosaz
"""
import sys
import errno
from DataSetCreate import DataSetCreate
from utils.constants import workdir 
from sklearn.metrics import accuracy_score
from DataSetPairCreate import DataSetPairCreate
from MiniAlexNet import MiniAlexNet
from EmbeddingNet import EmbeddingNet
from torch.utils.data import DataLoader
from utils.constants import workdir 
from test_siamese import test_siamese
from test_siamese_class import test_siamese_class
from test_siamese_diff import test_siamese_diff
from test_margine_dynamik import test_margine_dynamik
from test_siamese_roc import test_siamese_roc
from test_siamese_margine import test_siamese_margine
from test_siamese_margine_double import test_siamese_margine_double
from prova import writeJsonModel, plotLoss,writeJsonAccuracy,creteFileJson,addKeyValue,readFileDataset, lengthDataset,controlFileCSV,addValueJsonModel
import torch
import numpy as np
from prova import readJson
from test_classifier import test_classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
##################################################################################
def test_model(directory, path, model, version, resize,batch_size, margine = None):
    
    try:
        index= path.find("\\")
        index= path.find("\\",index+1)
        key1 = path[index+1:len(path)-4]
        print("key1",key1)
    except:
        key1="PerformanceTest"
        
    checkpoint = torch.load(path)
    siamese_test = model
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #siamese_test.to(device)
    checkpoint = torch.load(path)
    siamese_test.load_state_dict(checkpoint['model_state_dict'])

    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_test = dataSetPair.pair_money_test
    pair_money_test_loader = DataLoader(pair_test, batch_size, num_workers=0)
            
    
    #------------------------TESTARE SU DATI DEL TEST
    print("Testing on Test set....")
    #pair_prediction, pair_label, timeTest = test_siamese(siamese_reload, pair_money_test_loader, margin=2 )
    pair_prediction, pair_label, timeTest = test_siamese(siamese_test, pair_money_test_loader, margine= margine)
    numSimilPredette = np.sum(pair_prediction==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction == 1)
    print("Num Dissimil predette",numDissimilPredette)
    
    numSimilReali = np.sum(pair_label == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label==1)
    print("Num Dissimil Reali",numDissimilReali)
    
        #calculate Accuracy
    print(pair_prediction[0:10])
    print(pair_label[0:10])
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction)
    print("Precision di test: %0.4f"% precisionTest)
    
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average=None)
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)
        
    
    #--------------------------------
        
    print("Classification Report")
    print(classification_report(pair_label, pair_prediction))
    
    cm = confusion_matrix(pair_label, pair_prediction)
    print("Matrice di confusione \n",cm)
    cm.sum(1).reshape(-1,1)
        
    cm=cm/cm.sum(1).reshape(-1,1) #il reshape serve a trasformare il vettore in un vettore colonna
    print("\n")
    print("Matrice di confusione normalizzata \n",cm)
    
    tnr, fpr, fnr, tpr = cm.ravel()
    print("\n")
    print("TNR:",tnr)
    print("FPR:",fpr)
    print("FNR:",fnr)
    print("TPR:",tpr)
    
    key=key1
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","TNR","FPR","FNR","TPR","Timetesting"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,tnr,fpr,fnr,tpr,timeTest]
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[4], value[4])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[5], value[5])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[6], value[6])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[7], value[7])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[8], value[8]) 
    
    
def test_model_class(directory, path,  version, resize,batch_size, margine = None):
    try:
        index= path.find("\\")
        index= path.find("\\",index+1)
        key1 = path[index+1:len(path)-4]
        print("key1",key1)
    except:
        key1="PerformanceTest"
    
    siamese_test = torch.load(path)
    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_test = dataSetPair.pair_money_test
    pair_money_test_loader = DataLoader(pair_test, batch_size, num_workers=0)
                
    #------------------------ TESTARE SU DATI DEL TEST -----------
    print("Testing on Test set....")
    #pair_prediction, pair_label, timeTest = test_siamese(siamese_reload, pair_money_test_loader, margin=2 )
    timeTest,pair_prediction, pair_label  = test_siamese_class(siamese_test, pair_money_test_loader, margine= margine)
    
    numSimilPredette = np.sum(pair_prediction==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction == 1)
    print("Num Dissimil predette",numDissimilPredette)
    numSimilReali = np.sum(pair_label == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label==1)
    print("Num Dissimil Reali",numDissimilReali)
    
        #calculate Accuracy
    print(pair_prediction[0:10])
    print(pair_label[0:10])
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction)
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average=None)
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)
        
    
    #--------------------------------
    
    print("Classification Report")
    print(classification_report(pair_label, pair_prediction))
    
    cm = confusion_matrix(pair_label, pair_prediction)
    print("Matrice di confusione \n",cm)
    cm.sum(1).reshape(-1,1)
        
    cm=cm/cm.sum(1).reshape(-1,1) #il reshape serve a trasformare il vettore in un vettore colonna
    print("\n")
    print("Matrice di confusione normalizzata \n",cm)
    
    tnr, fpr, fnr, tpr = cm.ravel()
    print("\n")
    print("TNR:",tnr)
    print("FPR:",fpr)
    print("FNR:",fnr)
    print("TPR:",tpr)
    
    key=key1
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","TNR","FPR","FNR","TPR","Timetesting"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,tnr,fpr,fnr,tpr,timeTest]
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[4], value[4])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[5], value[5])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[6], value[6])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[7], value[7])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[8], value[8]) 

def test_model_margine(directory, path,  version, resize,batch_size, margine = 1.0):
    
    try:
        index= path.find("\\")
        index= path.find("\\",index+1)
        key1 = path[index+1:len(path)-4]
        print("key1",key1)
    except:
        key1="PerformanceTest"
    
    soglia = margine
    print("Soglia ",soglia)
    siamese_test = torch.load(path)
    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_test = dataSetPair.pair_money_test
    pair_money_test_loader = DataLoader(pair_test, batch_size, num_workers=0)
                
    #------------------------TESTARE SU DATI DEL TEST-----------
    print("Testing on Test set....")
    #pair_prediction, pair_label, timeTest = test_siamese(siamese_reload, pair_money_test_loader, margin=2 )
    timeTest,pair_prediction, pair_label  = test_siamese_margine(siamese_test, pair_money_test_loader,  soglia)
    
    numSimilPredette = np.sum(pair_prediction==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction == 1)
    print("Num Dissimil predette",numDissimilPredette)
    numSimilReali = np.sum(pair_label == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label==1)
    print("Num Dissimil Reali",numDissimilReali)
    
        #calculate Accuracy
    print(pair_prediction[0:10])
    print(pair_label[0:10])
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction)
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average=None)
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)
        
    
    #--------------------------------
    
    #key=["accuracy","precision","recall","mf1_score","time"]

    print("Classification Report")
    print(classification_report(pair_label, pair_prediction))
    
    cm = confusion_matrix(pair_label, pair_prediction)
    print("Matrice di confusione \n",cm)

    cm.sum(1).reshape(-1,1)
    cm=cm/cm.sum(1).reshape(-1,1) #il reshape serve a trasformare il vettore in un vettore colonna
    print("\n")
    print("Matrice di confusione normalizzata \n",cm)
    
    
    tnr, fpr, fnr, tpr = cm.ravel()
    print("\n")
    print("TNR:",tnr)
    print("FPR:",fpr)
    print("FNR:",fnr)
    print("TPR:",tpr)
    key=key1
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","TNR","FPR","FNR","TPR","Timetesting"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,tnr,fpr,fnr,tpr,timeTest]
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[4], value[4])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[5], value[5])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[6], value[6])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[7], value[7])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[8], value[8])  
#-------------------------- MODEL 6 v4 MARGINE DOUBLE---------------
def test_model_margine_double(directory, path,  version, resize,batch_size, margin1,margin2):
    
    try:
        index= path.find("\\")
        index= path.find("\\",index+1)
        key1 = path[index+1:len(path)-4]
        print("key1",key1)
    except:
        key1="PerformanceTest"
    
    print("version", version)
    print("key1",key1)
    print("Margine_1 ",margin1)
    print("Margine_2 ",margin2)
    siamese_test = torch.load(path)
    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_test = dataSetPair.pair_money_test
    pair_money_test_loader = DataLoader(pair_test, batch_size, num_workers=0)
                
    #------------------------TESTARE SU DATI DEL TEST-----------
    print("Testing on Test set....")
    #pair_prediction, pair_label, timeTest = test_siamese(siamese_reload, pair_money_test_loader, margin=2 )
    timeTest,pair_prediction, pair_label  = test_siamese_margine_double(siamese_test, pair_money_test_loader,  margin1,margin2)
    
    numSimilPredette = np.sum(pair_prediction==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction == 1)
    print("Num Dissimil predette",numDissimilPredette)
    numSimilReali = np.sum(pair_label == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label==1)
    print("Num Dissimil Reali",numDissimilReali)
    
        #calculate Accuracy
    print(pair_prediction[0:10])
    print(pair_label[0:10])
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction)
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average=None)
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)
        
    
    #--------------------------------
    

    print("Classification Report")
    print(classification_report(pair_label, pair_prediction))
    
    cm = confusion_matrix(pair_label, pair_prediction)
    print("Matrice di confusione \n",cm)

    cm.sum(1).reshape(-1,1)
    cm=cm/cm.sum(1).reshape(-1,1) #il reshape serve a trasformare il vettore in un vettore colonna
    print("\n")
    print("Matrice di confusione normalizzata \n",cm)
    
    
    tnr, fpr, fnr, tpr = cm.ravel()
    print("\n")
    print("TNR:",tnr)
    print("FPR:",fpr)
    print("FNR:",fnr)
    print("TPR:",tpr)
    
    key=key1
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","TNR","FPR","FNR","TPR","Timetesting"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,tnr,fpr,fnr,tpr,timeTest]
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[4], value[4])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[5], value[5])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[6], value[6])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[7], value[7])
    addValueJsonModel(directory+"modelTrained.json",version, key ,entry[8], value[8])
    
#########################-------- MODEL 6----------------
def test_model_margine_dynamik(directory,path, version,resize,batch_size, margine=None):
    
    siamese_test = torch.load(path)
    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_test = dataSetPair.pair_money_test
    pair_money_test_loader = DataLoader(pair_test, batch_size, num_workers=0)
    percorso = directory+"modelTrained.json"
    
    soglia = readJson(percorso,version,"euclidean_distance_threshold","last")
    #------------------------ TESTARE SU DATI DEL TEST -----------
    print("Testing on Test set....")
    #pair_prediction, pair_label, timeTest = test_siamese(siamese_reload, pair_money_test_loader, margin=2 )
    timeTest,pair_prediction, pair_label  = test_margine_dynamik(siamese_test, pair_money_test_loader,soglia, margine= margine)
    
    numSimilPredette = np.sum(pair_prediction==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction == 1)
    print("Num Dissimil predette",numDissimilPredette)
    numSimilReali = np.sum(pair_label == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label==1)
    print("Num Dissimil Reali",numDissimilReali)
    
        #calculate Accuracy
    print(pair_prediction[0:10])
    print(pair_label[0:10])
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction)
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    if recallTest!= 0.0 and precisionTest != 0.0:
        
        scores_testing = f1_score(pair_label,pair_prediction, average=None)
        scores_testing = scores_testing.mean()
        print("mF1 score di testing: %0.4f"% scores_testing)
        
        
    else:
        scores_testing = 0.000
        print("mscoref1",scores_testing)
        
    
    #--------------------------------
    
    key=["accuracy","precision","recall","mf1_score","time"]
    entry=["accuracyTest","precisionTest","recallTest","f1_score_Test","testing"]
    value=[accuracyTest,precisionTest,recallTest,scores_testing,timeTest]
    addValueJsonModel(directory+"modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key[4] ,entry[4], value[4])

def test_model_performance(directory, path,  version, resize,batch_size):
    
    
    siamese_test = torch.load(path)
    controlFileCSV()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
    pair_train = dataSetPair.pair_money_train
    pair_val = dataSetPair.pair_money_val
    pair_money_train_loader = DataLoader(pair_train, batch_size, num_workers=0)
    pair_money_valid_loader = DataLoader(pair_val, batch_size, num_workers=0)
    
    
    #------------------------TESTARE SU DATI DEL TEST-----------
    print("Performance....")
    #pair_prediction, pair_label, timeTest = test_siamese(siamese_reload, pair_money_test_loader, margin=2 )
    test_siamese_roc(siamese_test, pair_money_train_loader, pair_money_valid_loader, directory,version)
    

def reload_model_2_v3():
    print("model 2 v3")
    
#########################-------- MODEL 3----------------
def reload_model_3_v1():
    print("model 3 v1")  

def reload_model_3_v2():
    print("model 3 v2")

def reload_model_3_v3():
    print("model 3 v3")

