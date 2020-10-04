# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:15:11 2020

@author: rosaz
"""
import sys
import errno
import numpy as np
from DataSetCreate import DataSetCreate
from utils.constants import workdir 
from train_siamese_class import train_siamese_class
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from DataSetPairCreate import DataSetPairCreate
from MiniAlexNet import MiniAlexNet
from EmbeddingNet import EmbeddingNet
from torch.utils.data import DataLoader
from train_siamese import train_siamese
from train_siamese_margin_double import train_siamese_margin_double
from modelTest import test_model_class, test_siamese_class
from train_siamese_diff import train_siamese_diff
from test_margine_dynamik import test_margine_dynamik
from train_margine_dynamik import train_margine_dynamik
from train_siamese_class_adam import train_siamese_class_adam
from train_siamese_distrib_margine import train_siamese_distrib_margine
from train_siamese_margine import train_siamese_margine
from test_siamese import test_siamese
from test_siamese_margine import test_siamese_margine
from gaussian_distribution import gaussian_distribition
from prova import writeJsonModelInit ,writeJsonModelInit1,addValueJsonModel,writeJsonModelClass,writeJsonModel, plotLoss,writeJsonAccuracy,creteFileJson,addKeyValue,readFileDataset, lengthDataset,controlFileCSV,controlFileCSVPair,createFolder,plotAccuracy
import torch
import warnings


#########################-------- MODEL 1----------------
def train_model(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize,modeLoss=None):
   
    warnings.filterwarnings('always')
    
    directory =directory
    version=version
    lr=lr
    epochs=epochs
    momentum=momentum
    batch_size = batch_size
    resize=resize
    controlFileCSV()
    controlFileCSVPair()
   
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
    #training
    #modello, tempo di training, loss su train, loss su val
    print("Training...")
   
    siamese_money, timeTraining, array_loss_train, array_loss_val, array_sample_train, array_sample_valid,array_acc_train, array_acc_valid,labels_train,prediction_train,labels_val,prediction_val = train_siamese(siamese_money, pair_money_train_loader, pair_money_val_loader, exp_name , lr=lr, epochs = epochs,modeLoss=modeLoss)
    
    print("time Training\n ", timeTraining)
    print("Loss last on train", array_loss_train[-1])
    print("Loss last on valid\n", array_loss_val[-1])
    
    print("Array sample last on train", array_sample_train[-1])
    print("Array sample on last  valid\n", array_sample_valid[-1])
       
    
    print("lunghezza array accuracy TRAIN",len(array_acc_train))
    print("Lunghezza array SAMPLE train",len(array_sample_train))
    
    print("lunghezza array accuracy VALID",len(array_acc_valid))
    print("Lunghezza array SAMPLE VALID",len(array_sample_valid))
    
    #controlla se è presente la directory, altrimenti la crei
    createFolder(directory)
    #plot 
    plotLoss(directory,filename,array_loss_train, array_loss_val, array_sample_train, array_sample_valid)
    plotAccuracy(directory,filename,array_acc_train,array_acc_valid,array_sample_train,array_sample_valid)
        
    #------------------------TESTARE SU DATI DEL TRAIN
    #device = "cuda" if torch.cuda.is_available() else "cpu" 
    #siamese_money.to(device)
    print("Score on dataTrain...")
    #pair_predictionTrain, pair_labelTrain , timeTrain = test_siamese(siamese_money, pair_money_train_loader, margin=2 )
        #calculate Accuracy
    accuracyTrain = accuracy_score(labels_train, prediction_train)
    print("Accuarcy di train of last batch: %0.4f"% accuracyTrain)
        #calculate Precision
    precisionTrain = precision_score(labels_train, prediction_train)
    print("Precision di of last batch train: %0.4f"% precisionTrain)
            #calculate Recall
    recallTrain = recall_score(labels_train, prediction_train)
    print("Recall of last batch di train: %0.4f"% recallTrain)
    
    if recallTrain !=0.0 and precisionTrain !=0.0 :
        #calculate F1 score
       scores_training = f1_score(labels_train, prediction_train, average=None)
       scores_training = scores_training.mean()
       print("F1 score of last bacth di train: %0.4f"% scores_training)
    else:
        scores_training = 0.000
        print("F1 score of last bacth di train: %0.4f"% scores_training)
    
    #------------------------TESTARE SU DATI DEL VALID
    print("Score on dataValid...")
    #pair_predictionValid, pair_labelValid , timeValid = test_siamese(siamese_money, pair_money_val_loader, margin=2 )
        #calculate Accuracy
    accuracyValid = accuracy_score(labels_val , prediction_val)
    print("Accuarcy di validation: %0.4f"% accuracyValid)
        #calculate Precision
    precisionValid = precision_score(labels_val , prediction_val)
    print("Precision di validation: %0.4f"% precisionValid)
        #calculate Recall
    recallValid = recall_score(labels_val , prediction_val)
    print("Recall di validation: %0.4f"% recallValid)
      #calculate F1 score
    if recallValid !=0.0 and recallTrain !=0.0:
        
        scores_valid = f1_score(labels_val , prediction_val, average=None)
        scores_valid = scores_valid.mean()
        print("mF1 score di validation: %0.4f"% scores_valid)
        
    else:
        scores_valid = 0.00
        print("mF1 score di validation: %0.4f" % scores_valid )
    
    """ QUESTO VA FATTO IN FASE DI TESTING UTILIZZANDO I DATI DEL TEST E IL COMANDO RELOAD 
    #------------------------TESTARE SU DATI DEL TEST
    print("Testing on dataTest....")
    pair_prediction, pair_label, timeTest = test_siamese(siamese_money, pair_money_test_loader, margin=2 )
        #calculate Accuracy
    accuracyTest = accuracy_score(pair_label, pair_prediction)
    print("Accuarcy di test: %0.4f"% accuracyTest)
        #calculate Precision
    precisionTest = precision_score(pair_label, pair_prediction)
    print("Precision di test: %0.4f"% precisionTest)
        #calculate Recall
    recallTest = recall_score(pair_label, pair_prediction)
    print("Recall di test: %0.4f"% recallTest)
        #calculate F1 score
    scores_testing = f1_score(pair_label,pair_prediction, average=None)
    print("F1 score di testing: %0.4f"% scores_testing)
    """
    
    
    #-------------------------
    
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    contrastiveLoss = {"lossTrain": array_loss_train[-1], "lossValid":array_loss_val[-1]}
    accuracy = {"accuracyTrain":accuracyTrain , "accuracyValid":accuracyValid }
    precision = {"precisionTrain":precisionTrain , "precisionValid":precisionValid }
    recall = {"recallTrain":recallTrain , "recallValid":recallValid }
    f1score = {"f1_score_Train":scores_training , "f1_score_Valid":scores_valid}
    time = {"training": timeTraining}
    
    
    writeJsonModel(directory,name,version, hyperparametr, batch_size, contrastiveLoss, accuracy ,f1score, precision, recall, time)
    
    #salvataggio su file json
    #writeJsonModel(path, num, array_loss_train[-1], array_loss_val[-1], len(pair_train),len(pair_validation), epochs, time)
    #writeJsonAccuracy(path, "accuracyTest", accuracy)

def train_model_class_v1(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize,decay=None,modeLoss=None, dizionario_array = None):
    
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
    
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version) 
    """?????"""
    
    #training
    #modello, tempo di training, loss su train, loss su val
    print("Training...")
   
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_class(directory,version,siamese_money, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name ,decay, lr=lr, epochs = epochs,momentum=momentum,logdir='logs', dizionario =None)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
  
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, contrastiveLoss, accuracy ,time)
    
    namep=exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_class(siamese_model, pair_money_val_loader, margine= modeLoss)
    
    numSimilPredette = np.sum(pair_prediction_val==0)
    print("Num Simili predette",numSimilPredette)
    numDissimilPredette = np.sum(pair_prediction_val == 1)
    print("Num Dissimil predette",numDissimilPredette)
    numSimilReali = np.sum(pair_label_val == 0)
    print("Num Simili Reali",numSimilReali)
    numDissimilReali = np.sum(pair_label_val==1)
    print("Num Dissimil Reali",numDissimilReali)
    print("Time testing on validation", timeVal)
        #calculate Accuracy
    print(pair_prediction_val[0:10])
    print(pair_label_val[0:10])
    accuracyVal = accuracy_score(pair_label_val, pair_prediction_val)
    print("Accuarcy di validation: %0.4f"% accuracyVal)
        #calculate Precision
    precisionVal = precision_score(pair_label_val, pair_prediction_val)
    print("Precision di validation: %0.4f"% precisionVal)
        #calculate Recall
    recallVal = recall_score(pair_label_val, pair_prediction_val)
    print("Recall di validation: %0.4f"% recallVal)
        #calculate F1 score
    if recallVal!= 0.0 and precisionVal != 0.0:
        
        scores_testing_val = f1_score(pair_label_val,pair_prediction_val, average=None)
        scores_testing_val = scores_testing_val.mean()
        print("mF1 score di validation: %0.4f"% scores_testing_val)
        
        
    else:
        scores_testing_val = 0.000
        print("mscoref1",scores_testing_val)
        
    
    key=["accuracy","precision","recall","mf1_score","time"]
    entry=["accuracyValid","precisionValid","recallValid","f1_score_Valid","testValid"]
    value=[accuracyVal,precisionVal,recallVal,scores_testing_val,timeVal]
    addValueJsonModel(directory+"\\modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"\\modelTrained.json",version, key[4] ,entry[4], value[4])
        
    
    # modello con 1 margine, etichetta è assegnata mediante una soglia
def train_model_margine(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize, decay=None, margin=None,soglia=None,modeLoss=None):
    # directory es "Model-6"
    createFolder(directory)
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
   
    
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version)
    
    print("Training...")
   
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_margine(directory,version,siamese_money, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name=exp_name , lr=lr, epochs = epochs,momentum=momentum,margin=margin,soglia=soglia,logdir='logs', decay=decay,modeLoss=modeLoss)
    
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
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,resize, batch_size, contrastiveLoss, accuracy ,time)
    
    namep=exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_margine(siamese_model, pair_money_val_loader, margine= modeLoss)
    
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
    entry=["accuracyVal","precisionVal","recallVal","f1_score_Val","timeVal"]
    value=[accuracyVal,precisionVal,recallVal,scores_testing_val,timeVal]
    addValueJsonModel(directory+"modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key[4] ,entry[4], value[4])

    # modello con 1 margine, etichetta è assegnata mediante una soglia
def train_model_margine_double(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize, decay=None, margin1=None,margin2=None, modeLoss=None):
    # directory es "Model-6"
    createFolder(directory)
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
   
    
    createFolder(directory+"\\"+version)
    createFolder(directory+"\\"+version+"\\"+"Metod2")

    
    writeJsonModelInit1(directory,name,version)
    
    print("Training...")
    
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_margin_double(directory,version,siamese_money, pair_money_train_loader, pair_money_val_loader,resize,batch_size, exp_name=exp_name , lr=lr, epochs = epochs,momentum=momentum,margin1=margin1,margin2=margin2, logdir='logs', decay=decay,modeLoss=modeLoss)
    
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
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,resize, batch_size, contrastiveLoss, accuracy ,time)
"""    
    namep=exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_margine(siamese_model, pair_money_val_loader, margine= modeLoss)
    
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
    entry=["accuracyVal","precisionVal","recallVal","f1_score_Val","timeVal"]
    value=[accuracyVal,precisionVal,recallVal,scores_testing_val,timeVal]
    addValueJsonModel(directory+"modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key[4] ,entry[4], value[4])
  """  
    # modello con 1 margine, etichetta è assegnata mediante una soglia
def train_model_margine_diff(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize,margin1, margin2, decay=None,modeLoss=None,dizionario_array=None):
    # directory es "Model-6"
    createFolder(directory)
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
   
    
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version)
    
    print("Training...")
   
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_diff(directory,version,siamese_money, pair_money_train_loader, pair_money_val_loader,resize,batch_size, margin1,margin2,exp_name=exp_name , lr=lr, epochs = epochs,momentum=momentum,logdir='logs', decay=decay,modeLoss=modeLoss, dizionario_array=dizionario_array)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    if not decay is None:
         
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr,"decay":decay, "momentum" : momentum,"margin1":margin1,"margin2":margin2, "numSampleTrain": len(pair_train) }
    else:  
        hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum,"margin1":margin1,"margin2":margin2, "numSampleTrain": len(pair_train) }
    
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,resize, batch_size, contrastiveLoss, accuracy ,time)
    
    namep=exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_margine(siamese_model, pair_money_val_loader, margine= modeLoss)
    
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
    entry=["accuracyVal","precisionVal","recallVal","f1_score_Val","timeVal"]
    value=[accuracyVal,precisionVal,recallVal,scores_testing_val,timeVal]
    addValueJsonModel(directory+"modelTrained.json",version, key[0] ,entry[0], value[0])
    addValueJsonModel(directory+"modelTrained.json",version, key[1] ,entry[1], value[1])
    addValueJsonModel(directory+"modelTrained.json",version, key[2] ,entry[2], value[2])
    addValueJsonModel(directory+"modelTrained.json",version, key[3] ,entry[3], value[3])
    addValueJsonModel(directory+"modelTrained.json",version, key[4] ,entry[4], value[4])


#########################-------- MODEL 5----------------
def train_model_class_adam(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize,modeLoss=None):
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
    #training
    #modello, tempo di training, loss su train, loss su val
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version)
    
    print("Training...")
   
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_class_adam(directory,version,siamese_money, pair_money_train_loader, pair_money_val_loader,resize, batch_size, exp_name , lr=lr, epochs = epochs,modeLoss=modeLoss)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, contrastiveLoss, accuracy ,time)
    
    
    namep=exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_class(siamese_model, pair_money_val_loader, margine= modeLoss)
    
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
    
    
#A------------------------------- MODEL 6 
def train_model_margine_dynamik(directory,filename,version,exp_name ,name, model, lr,epochs,momentum,batch_size,resize):
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    
    #training
    #modello, tempo di training, loss su train, loss su val
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version)
    
    print("Training...")
   
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_margine_dynamik(directory,version,model, pair_money_train_loader, pair_money_val_loader,resize, batch_size, exp_name , lr=lr, epochs = epochs)
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, contrastiveLoss, accuracy ,time)
    
    
    namep= exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_margine_dynamik(siamese_model, pair_money_val_loader)
    
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
    
         

def distribution(directory,version, model, batch_size,resize):
    print("Calculate mean of distribution")
    #• directory = "Model-7"
    createFolder(directory)
    createFolder(directory+"\\"+version)
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    gaussian_distribition(directory, version, model, pair_money_train_loader, pair_money_val_loader,pair_money_test_loader,resize,batch_size, exp_name='model_1')
    
    

#######################--------- MODEL 7 v2 ----------------

def train_model_margine_distrib(directory,filename,version,exp_name,name, model,lr,epochs,momentum,batch_size,resize,margine,decay=None,modeLoss=None, dizionario_array = None):
    
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    pair_train = dataSetPair.pair_money_train
    pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    pair_money_test_loader = DataLoader(pair_test, batch_size=batch_size, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
    siamese_money = model # modello
    
    createFolder(directory+"\\"+version)
    writeJsonModelInit1(directory,name,version) 
    """?????"""
    
    #training
    #modello, tempo di training, loss su train, loss su val
    print("Training...")
   
    modello ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val = train_siamese_distrib_margine(directory,version,siamese_money, pair_money_train_loader, pair_money_val_loader,resize,batch_size, margine,exp_name , decay, lr=lr, epochs = epochs,momentum=momentum,logdir='logs', modeLoss="single")
    
    print("Time computing", f)
    print("last_loss_train",last_loss_train)
    print("last_loss_val",last_loss_val)
    print("last_acc_train",last_acc_train)
    print("last_acc_val",last_acc_val)
    
    hyperparametr = {"indexEpoch":epochs-1,"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": f}
    
    
  
    writeJsonModelClass(directory,name,version, hyperparametr,resize,batch_size, contrastiveLoss, accuracy ,time)
    
    namep= exp_name+".pth"
    siamese_model = torch.load(namep)
    
    print("Testing on Validation set")
    
    timeVal,pair_prediction_val, pair_label_val  = test_siamese_class(siamese_model, pair_money_val_loader, margine= modeLoss)
    
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
             

def train_model_3_v2():
    print("model 3 v2")

def train_model_3_v3():
    print("model 3 v3")
    




    