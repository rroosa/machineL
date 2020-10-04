# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:11:45 2020

@author: rosaz
"""

import sys
import errno
import os
import json
from matplotlib import pyplot as plt
from utils.constants import workdir 
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils.calculate_time import Timer
from time import time
from timeit import default_timer as timer


def controlFileCSVPair():
    listFile = []
    try:
        with open(workdir+"Pair_train.csv") as f:
                listFile.append("Pair_train.csv")
        with open(workdir+"Pair_validation.csv") as f:
                listFile.append("Pair_validation.csv")
        with open(workdir+"Pair_test.csv") as f:
                listFile.append("Pair_test.csv")
                # File exists
        print("Files Pair_train | Pair_validation | Pair_test  exist\n")
    except IOError as e:
        sys.stderr.write("fatal error, try run --create datasetPair")
        exit()
            # Raise the exception if it is not ENOENT (No such file or directory)
        if e.errno != errno.ENOENT:
            sys.stderr.write("fatal error")
            exit(0) 
            
def controlFileCSVBase():
    listFile = []
    try:
        with open(workdir+"train_base.csv") as f:
                listFile.append("train_base.csv")
        with open(workdir+"valid_base.csv") as f:
                listFile.append("valid_base.csv")
        with open(workdir+"test_base.csv") as f:
                listFile.append("test_base.csv")
                # File exists
        print("Files train | valid | test  exist\n")
    except IOError as e:
        sys.stderr.write("fatal error, try run --create datasetBase")
        exit()
            # Raise the exception if it is not ENOENT (No such file or directory)
        if e.errno != errno.ENOENT:
            sys.stderr.write("fatal error")
            exit(0)
            
def controlFileCSV():
    listFile = []
    try:
        with open(workdir+"train.csv") as f:
                listFile.append("train.csv")
        with open(workdir+"valid.csv") as f:
                listFile.append("valid.csv")
        with open(workdir+"test.csv") as f:
                listFile.append("test.csv")
                # File exists
        print("Files train | valid | test  exist\n")
    except IOError as e:
        sys.stderr.write("fatal error, try run --create dataset")
        exit()
            # Raise the exception if it is not ENOENT (No such file or directory)
        if e.errno != errno.ENOENT:
            sys.stderr.write("fatal error")
            exit(0)
            
def controlFile():
   try:
    with open("Dataset/train.csv") as f:
        print("ok")
    with open("Dataset/valid.csv") as f:
        print("ok")
    with open("Dataset/test.csv") as f:
        print("ok")
        # File exists
   except IOError as e:
    sys.stderr.write("fatal error")
    exit()
    # Raise the exception if it is not ENOENT (No such file or directory)
    if e.errno != errno.ENOENT:
        sys.stderr.write("fatal error")
        exit(0)
        
        print("ciao")

"""FUNZIONA controlla se esiste una cartella"""
def controlFolder(directory):
    
    if not os.path.exists(directory):
        sys.stderr.write("Error: folder '%s' not exists, crete your dataset" % directory)
        return
          


"""FUNZIONA crea o controlla la cartella"""
def createFolder(path):
    access_rights = 0o777
    try:
            if not os.path.exists(path):
                os.mkdir(path,access_rights)
            
    except OSError:
            print("Creation of the directory %s failed" % path)
    else:
            print("Directory %s was successfully created or is already present" % path)

def writeJsonAccuracy(path, fileName, entry, accuracy, entryTime, time):
    #a_cc = {entry: accuracy}
    #timeTrain = {entryTime: time}
    
    # se il file non esistw crealo nuovo e scrivi l'oggetto 
    createFolder(path)
    if not os.path.exists(path+"\\"+fileName):
         #print("File non esiste")
         entry = {entry: accuracy, entryTime: time}
         with open(path+"\\"+fileName,"w") as outfile:
             json.dump(entry,outfile) 
             
 
    #altrimenti se il file esiste 
    #prova a fare il parsing
    else:
        #print("Qui")
        try:
    # Read in the JSON document,  parsing è stato effettuato con successo
            with open(path+"\\"+fileName,"r") as outfile:
                #print("qui3")
                datum = json.load(outfile)
                
                # modifica il valore della chiave se esiste
            if not (datum.get(entry) is None):
                #print("value is present for given JSON key")
                print(datum.get(entry))
                datum[entry]=accuracy
                with open(path+"\\"+fileName, "w") as outfile:
                    json.dump(datum, outfile)
            else:
                #print("Chiave non esiste")
                    #entry = {entry: accuracy, entryTime: time}
                datum[entry]=accuracy
                with open(path+"\\"+fileName, "w") as json_outfile:
                    json.dump(datum, json_outfile)
                    
            if not (datum.get(entryTime) is None):
                #print("value is present for given JSON key")
                print(datum.get(entryTime))
                datum[entryTime]=time
                with open(path+"\\"+fileName, "w") as json_outfile:
                    json.dump(datum, json_outfile)
            else:
                #print("Chiave non esiste")
                
                datum[entryTime]=time
                with open(path+"\\"+fileName, "w") as json_outfile:
                    json.dump(datum,json_outfile)
                
        except:
            #print("Qui2")
            entry = {entry: accuracy, entryTime: time}
            with open(path+"\\"+fileName, "w") as outfile:
                 json.dump(entry,outfile)
    
"""FUNZIONA creazione file"""
def creteFileJson(path, entry):
    try:
        if not os.path.exists(path):
            
            with open(path, "w") as outfile:
                json.dump(entry,outfile,indent=2)
    except:
        sys.stderr.write("Error create file")
        exit(0)

def writeJsonNorma(path,media,dev,time):
    
    media = media.tolist()
    dev = dev.tolist()
    if not os.path.exists(path):
        entry={"normalize":{"mean":media,"dev_std":dev,"computeTime":time}}
        with open(path, "w") as outfile:
            json.dump(entry,outfile,indent=2)
    else:
            
        with open(path, "r") as outfile:
            data=json.load(outfile)
        
        if not (data.get("normalize") is None):
            #print("value is present for given JSON key")
            #print(data.get("normalize"))
            #aggiungi chiavi
            entry ={"mean":media,"dev_std":dev,"computeTime":time}
            data["normalize"]=entry
            with open(path, "w") as outfile:
                json.dump(data,outfile,indent=2)
        else:
            entry ={"mean":media,"dev_std":dev,"computeTime":time}
            data["normalize"]=entry

            with open(path, "w") as outfile:
                json.dump(data,outfile,indent=2)
                
def addKey(path,key,obj):
    with open(path,"r") as json_file:
        data = json.load(json_file)
    if not(data.get(key) is None):
        data[key]=obj
        with open(path, "w") as json_file:
            json.dump(data,json_file, indent=2)
    else:
        data[key]=obj
        with open(path, "w") as json_file:
            json.dump(data,json_file, indent=2)
        
    
    
"""FUNZIONA"""
def addKeyValue(path,key,value):
    with open(path, "r") as json_file:
        data = json.load(json_file)
        
    if not (data.get(key) is None):
        #print(data.get(key))
        data[key]=value   
        with open(path, "w") as json_file:
            json.dump(data, json_file,indent=2)
    else:
        data[key]=value   
        with open(path, "w") as json_file:
            json.dump(data, json_file,indent=2)

"""FUNZIONA"""
def readJson(path,version,entry,key):
    with open(path, "r") as json_file:
        data = json.load(json_file)
        #print(data)
    if not (data.get(version) is None):
        obj = data[version]  
        #print(obj)
        if not (obj.get(entry) is None):
            entryObj = obj[entry]
            
            if not (entryObj.get(key) is None):
                value = entryObj[key]
                return value
            else:
                return None


            
    
    
    

def writeJsonNorm(path,media,dev,time):
    media = media.tolist()
    dev = dev.tolist()
    
    #entry={"normalize":{"mean":media,"dev_std":dev,"time":time}}
    with open(path, "r") as outfile:
        data=json.load(outfile)
        
    if not (data.get(media) is None):
        #print("value is present for given JSON key")
        print(data.get(media))
        #aggiungi chiavi
        data["mean"]=media

        with open(path, "w") as outfile:
            json.dump(data,outfile)
    else:
        data["mean"]=media

        with open(path, "w") as outfile:
            json.dump(data,outfile)
        
            
    if not (data.get(dev) is None):
        #print("value is present for given JSON key")
        print(data.get(dev))
        #aggiungi chiavi
        data["dev_std"]=dev
        with open(path, "w") as outfile:
            json.dump(data,outfile)
    else:
        data["dev_std"]=dev
        with open(path, "w") as outfile:
            json.dump(data,outfile)
            
    if not (data.get(time) is None):
        #print("value is present for given JSON key")
        print(data.get(time))
        #aggiungi chiavi
        data["time"] = time
        with open(path, "w") as outfile:
            json.dump(data,outfile)
    else:
        data["time"] = time
        with open(path, "w") as outfile:
            json.dump(data,outfile)
    

def writeJson(model,num, media, dev, time):
    media = media.tolist()
    dev = dev.tolist()
    path = model
    access_rights = 0o777
    try:
        if not os.path.exists(path):
            os.mkdir(path,access_rights)
            print("Successfully created the directory %s" % path)
        else:
            print("Directory exist")
        
    except OSError:
           print("Creation of the directory %s failed" % path)
           exit(0)
           

    data = {"model":num, "mean":media, "dev_std":dev, "time":time}
    with open(path+"\\normalize.json", "w") as outfile:
        json.dump(data,outfile)
    # No such file or directory

"""FUNZIONA"""
def writeJsonModel(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,f1score, precision, recall, time):
    
    access_rights = 0o777
    try:   # se la directory non è presente, creare
        if not os.path.exists(directory):
            os.mkdir(directory,access_rights)
            print("Successfully created the directory %s" % directory)
        else:
            print("Save file in Directory: %s " % directory)
            
    except OSError:
           sys.stderr.write("Creation of the directory %s failed" % directory)
           exit(0)
           
    value= {"hyperparametr":hyperparametr,"resize":resize,"dimBatch":batch_size, "contrastiveLoss":contrastiveLoss,"accuracy":accuracy,"precision":precision, "recall":recall,"mf1_score":f1score,"time":time}
    
    # se il file non esiste scrivilo nuovo
    if not os.path.exists(directory+"\\modelTrained.json"):
        datawrite = {"EmbeddingNet":name, version:value}
        
        with open(directory+"\\modelTrained.json", "w") as outfile:
             json.dump(datawrite,outfile,indent=2)
    
    else:# se il file esiste, 
        #controlla se la versione del modello esiste già
        with open(directory+"\\modelTrained.json", "r") as outfile:
            data = json.load(outfile)
        if not (data.get(version) is None): # se la versione esiste gia, aggiorna i campi
            data[version]=value #aggiorna campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
        else:
            data[version] = value # aggiungi nuovo campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
            
        
    """
    with open(path+"\\modelTrained.json", "r") as outfile:      
    data = {"model":num, "lossTrain":lossTrain, "lossValidation":lossVal, "numPairTrain":numPairTrain, "numPairVal":numPairVal, "numEpochs":numEpoche, "timeTrain":time}
    with open(path+"\\modelTrained.json", "w") as outfile:
        json.dump(data,outfile)
    # No such file or directory
    """
   
"""FUNZIONA"""
def writeJsonModelInit(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time):
    
    access_rights = 0o777
    try:   # se la directory non è presente, creare
        if not os.path.exists(directory):
            os.mkdir(directory,access_rights)
            print("Successfully created the directory %s" % directory)
        else:
            print("Save file in Directory: %s " % directory)
            
    except OSError:
           sys.stderr.write("Creation of the directory %s failed" % directory)
           exit(0)
           
    value= {"hyperparametr":hyperparametr,"resize":resize,"dimBatch":batch_size, "contrastiveLoss":contrastiveLoss,"accuracy":accuracy,"time":time}
    
    # se il file non esiste scrivilo nuovo
    if not os.path.exists(directory+"\\modelTrained.json"):
        datawrite = {"EmbeddingNet":name, version:value}
        
        with open(directory+"\\modelTrained.json", "w") as outfile:
             json.dump(datawrite,outfile,indent=2)
    
    else:# se il file esiste, 
        #controlla se la versione del modello esiste già
        with open(directory+"\\modelTrained.json", "r") as outfile:
            data = json.load(outfile)
        if not (data.get(version) is None): # se la versione esiste gia, aggiorna i campi
            data[version]=value #aggiorna campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
        else:
            data[version] = value # aggiungi nuovo campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
                
    
"""FUNZIONA"""
def writeJsonModelInit1(directory,name,version):
    
    access_rights = 0o777
    try:   # se la directory non è presente, creare
        if not os.path.exists(directory):
            os.mkdir(directory,access_rights)
            print("Successfully created the directory %s" % directory)
        else:
            print("Save file in Directory: %s " % directory)
            
    except OSError:
           sys.stderr.write("Creation of the directory %s failed" % directory)
           exit(0)
           
    #value= {"hyperparametr":hyperparametr,"resize":resize,"dimBatch":batch_size, "contrastiveLoss":contrastiveLoss,"accuracy":accuracy,"time":time}
    
    # se il file non esiste scrivilo nuovo
    if not os.path.exists(directory+"\\modelTrained.json"):
        datawrite = {"EmbeddingNet":name, version:value}
        
        with open(directory+"\\modelTrained.json", "w") as outfile:
             json.dump(datawrite,outfile,indent=2)

"""FUNZIONA"""
def writeJsonModelEpoca(directory,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time):
              
    value= {"hyperparametr":hyperparametr,"resize":resize,"dimBatch":batch_size, "contrastiveLoss":contrastiveLoss,"accuracy":accuracy,"time":time}
 
    with open(directory+"\\modelTrained.json", "r") as outfile:
        data = json.load(outfile)
    if not (data.get(version) is None): # se la versione esiste gia, aggiorna i campi
            data[version]=value #aggiorna campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
    else:
            data[version] = value # aggiungi nuovo campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
                
    
def saveArray(directory,version,array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train,array_glb_valid,soglie=None):
    createFolder(directory+"\\"+version) 
    if not soglie is None:
         datawrite = {"array_loss_train":array_loss_train, "array_loss_valid":array_loss_valid, "array_accuracy_train":array_accuracy_train, "array_accuracy_valid":array_accuracy_valid, "array_glb_train":array_glb_train, "array_glb_valid":array_glb_valid, "euclidean_distance_threshold":soglie}
        
    else:
        datawrite = {"array_loss_train":array_loss_train, "array_loss_valid":array_loss_valid, "array_accuracy_train":array_accuracy_train, "array_accuracy_valid":array_accuracy_valid, "array_glb_train":array_glb_train, "array_glb_valid":array_glb_valid}
        
    with open(directory+"\\"+version+"\\value_arrays.json", "w") as outfile:
             json.dump(datawrite,outfile,indent=2)
     
def saveArray_metod2(directory,version,array_acc, array_f1 , array_recall, array_precision,tp,fp, array_glb_valid):
    createFolder(directory+"\\"+version+"\\"+"Metod2") 

    datawrite = {"array_acc_valid":array_acc, "array_f1_valid":array_f1, "array_recall_valid":array_recall, "array_precision_valid":array_precision, "array_tp_valid":tp, "array_fp_valid":fp, "gl_step_valid":array_glb_valid}
        
    with open(directory+"\\"+version+"\\"+"Metod2"+"\\"+"value_performance.json", "w") as outfile:
             json.dump(datawrite,outfile,indent=2)
       
"""FUNZIONA"""
def writeJsonModelClass(directory,name,version, hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time):
    
    access_rights = 0o777
    try:   # se la directory non è presente, creare
        if not os.path.exists(directory):
            os.mkdir(directory,access_rights)
            print("Successfully created the directory %s" % directory)
        else:
            print("Save file in Directory: %s " % directory)
            
    except OSError:
           sys.stderr.write("Creation of the directory %s failed" % directory)
           exit(0)
           
    value= {"hyperparametr":hyperparametr,"resize":resize,"dimBatch":batch_size, "loss":contrastiveLoss,"accuracy":accuracy,"time":time}
    
    # se il file non esiste scrivilo nuovo
    if not os.path.exists(directory+"\\modelTrained.json"):
        datawrite = {"EmbeddingNet":name, version:value}
        
        with open(directory+"\\modelTrained.json", "w") as outfile:
             json.dump(datawrite,outfile,indent=2)
    
    else:# se il file esiste, 
        #controlla se la versione del modello esiste già
        with open(directory+"\\modelTrained.json", "r") as outfile:
            data = json.load(outfile)
        if not (data.get(version) is None): # se la versione esiste gia, aggiorna i campi
            data[version]=value #aggiorna campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
        else:
            data[version] = value # aggiungi nuovo campo
            with open(directory+"\\modelTrained.json", "w") as outfile:
                json.dump(data,outfile,indent=2)
                
    
    
"""FUNZIONA"""

def addValueJsonModel(path,num, key ,entry, value):
    
    
    if not os.path.exists(path):
        print("File %s not is exists" % path)
        sys.stderr.write("File %s not is exists" % path)
        exit(0)
    else:
        #leggi il file
        with open(path, "r") as outfile:
            data = json.load(outfile)
        if not (data.get(num) is None): # se la versione esiste gia, aggiorna i campi
            #print(data.get(num))
            versione = data[num]
            if not (versione.get(key) is None): # se la chiave accuracy esiste, aggiorna i campi
                obj = versione[key]
                if not (obj.get(entry) is None):
                    obj[entry]=value
                    with open(path, "w") as outfile:
                        json.dump(data,outfile,indent=2)
                else:
                    obj[entry]=value
                    with open(path, "w") as outfile:
                        json.dump(data,outfile,indent=2)
            else:
                #print("non esiste")
                obj={entry:value}
                versione[key]=obj

                with open(path, "w") as outfile:
                    json.dump(data,outfile,indent=2)
        else:
            print("Add version %s" %num)
            dato={key:{entry:value}}
            data[num]=dato
            with open(path, "w") as outfile:
                json.dump(data,outfile,indent=2)
                    
"""FUNZIONA """
def writeJsonMargin(path,key,value):
    
    #entry={"normalize":{"mean":media,"dev_std":dev,"time":time}}
    with open(path, "r") as outfile:
        data=json.load(outfile)
        
    if not (data.get(key) is None):
        #print("value is present for given JSON key")
        print(data.get(key))
        #aggiungi chiavi
        data[key]=value

        with open(path, "w") as outfile:
            json.dump(data,outfile,indent=2)
    else:
        data[key]=value

        with open(path, "w") as outfile:
            json.dump(data,outfile, indent=2)






def readNorm(path):
    if not os.path.exists(path+'\\normalize.json'):
        sys.stderr.write("fatal error")
        exit(0)
    else:
        with open(path+'\\normalize.json') as json_file:
            data = json.load(json_file)
            if not (data.get("mean") is None):
                print("mean is present for given JSON key")
                arrayMean= data['mean']
                print(arrayMean)
            else:
                sys.stderr.write("fatal error-mean not exists")
                exit(0)
            if not (data.get("dev_std") is None):
                arrayDev= data['dev_std']
            else:
                sys.stderr.write("fatal error- dev_std not exists")
                exit(0)
                
            return arrayMean, arrayDev

def readFileDataset(path,entry):
    if not os.path.exists(path):
        sys.stderr.write("Dataset is not present, try --create dataset | datasetPair")
        exit(0)
    else:
        with open(path, "r") as outfile:
            data= json.load(outfile)
        if not (data.get(entry) is None):
            value = data[entry]
            return value
        else:
            sys.stderr.write("Dataset is not present, try --create dataset | datasetPair")
            exit(0)

def lengthDataset(path,entry,key):
    somma = 0
    if not os.path.exists(path):
        sys.stderr.write("Dataset is not present, try --create dataset")
        exit(0)
    else:
        with open(path, "r") as outfile:
            data= json.load(outfile)
        if not (data.get(entry) is None):
            value = data[entry]
            for obj in value:
                if not (obj.get(key) is None):
                    num = obj[key]
                    somma = somma + num
            return somma           
    
def calculateScore(labels,prediction):
    
    accuracy = accuracy_score(labels, prediction)
        #calculate Precision
    precision = precision_score(labels, prediction)
            #calculate Recall
    recall= recall_score(labels, prediction)
   
    if recall !=0 and precision !=0 :
        #calculate F1 score
       scores = f1_score(labels, prediction, average=None)
       scores = scores.mean()
       
    else:
        scores = 0.000
       
    return [accuracy, precision, recall, scores]


def controlNormalize(path):
   try:
    with open(path+"\\normalize.json") as f:
        response = input("Do you want to re-calculate the mean and standard deviation? y | n : ")
        if(response =="y"):
            print("recalculate")
            
            
        elif (response =="n"):
            print("no")
        else:
            controlNormalize()
           # File exists
   except IOError as e:
    print("Normalize")
    
    # Raise the exception if it is not ENOENT (No such file or directory)
    if e.errno != errno.ENOENT:
        sys.stderr.write("fatal error")
        exit(0)

def norm(im):
        im = im-im.min()
        return im/im.max() 

"""Funziona"""
           
def plotLoss(path,namefile,array_loss_train, array_loss_val, array_sample_train, array_sample_valid):
    plt.figure()
    plt.subplot(121)
    plt.ylabel('loss train')
    plt.xlabel('num samples')
    plt.grid()
    plt.plot( array_sample_train, array_loss_train)
    plt.subplot(122)
    
    plt.ylabel('loss validation')
    plt.xlabel('num samples')
    plt.grid()
    plt.plot(array_sample_valid, array_loss_val)
    plt.savefig(path+namefile+'plotLoss.png')
    plt.show()
              
def plotAccuracy(path,namefile,array_acc_train,array_acc_valid,array_sample_train,array_sample_valid):
    plt.figure()
    plt.subplot(121)
    plt.ylabel('Acc train')
    plt.xlabel('num samples')
    plt.grid()
    plt.plot( array_sample_train, array_acc_train)
    
    plt.subplot(122)   
    plt.ylabel('Acc validation')
    plt.xlabel('num samples')
    plt.grid()
    plt.plot(array_sample_valid, array_acc_valid)
    plt.savefig(path+namefile+'plotAcc.png')
    plt.show()
  
def net_save(epoch, net, optimizer, lossTrain,lossValid, acc_train ,acc_valid,global_step_train, global_step_valid, path,dict_stato_no = None):
    if not dict_stato_no is None:
        net = net.state_dict()
        
    torch.save({
            'epoch': epoch,
            'model_state_dict': net,
            'optimizer_state_dict': optimizer.state_dict(),
            'lossTrain': lossTrain,
            'lossValid': lossValid,
            'global_step_train':global_step_train,
            'global_step_valid':global_step_valid,
            'accTrain': acc_train,
            'accValid':acc_valid
            
            }, path)    
    
    

directory = "Model_1"

lr=0.01
epochs=20
momentum=0.99
batch_size = 15000
array_loss_train=[2,3,1,2,3,4]
array_loss_val =[2,3,9,0,0,7]
accuracyTrain=29891.2
accuracyValid=213131.22
precisionTrain=2
precisionValid=233
recallTrain=223
recallValid=32323
scores_training=2133,33
scores_valid=8799,3
timeTraining="323sss sec"
timeTrain="23233 sec"
timeValid="32332 sec"
pair_train=[2,3,2,1]
directory="Model-1"
name="LeNet2"
version="3"
epochs=20


hyperparametr = {"lr":lr, "momentum" : momentum, "numSampleTrain": len(pair_train) }
contrastiveLoss = {"lossTrain": array_loss_train[-1], "lossTest":array_loss_val[-1]}
accuracy = {"accuracyTrain":accuracyTrain , "accuracyValid":accuracyValid }
precision = {"precisionTrain":precisionTrain , "precisionValid":precisionValid }
recall = {"recallTrain":recallTrain , "recallValid":recallValid }
f1score = {"f1_score_Train":scores_training , "f1_score_Valid":scores_valid}
time = {"traning": timeTraining, "test_on_Train":timeTrain, "test_on_Valid": timeValid}

""" FUNZIONA"""
#writeJsonModel(directory,name,version, hyperparametr, epochs , batch_size, contrastiveLoss, accuracy ,f1score, precision, recall, time)    

path="Model-1"
array_loss_train=[1,2,3,4,5,6,7,8,9]
array_loss_val=[2,3,4,5,6,7,8,9,10]
array_sample_train=[10,20,30,40,50,60,70,80,90]
array_sample_valid=[20,30,40,50,60,70,80,90,100]
"""FUNZIONA"""
#plotLoss(path,array_loss_train, array_loss_val, array_sample_train, array_sample_valid)

key="time"
entry="timeTrain"
value="ValoreTime"

"""FUNZIONA"""
#addValueJsonModel("Model-1/modelTrained.json","1", key ,entry, value)
directory="Model-1"
"""FUNZIONA"""
#createFolder(directory)

"""FUNZIONA """
#controlFolder("Dataset")

"""FUNZIONA"""
#value = readJson("Model-1/ModelTrained.json","1","time","training")
#if not value is None:
#   print(type(value))
#   num= float(value)
#   print(type(num))
#   num= num+ 5.8
#   print(num)
   
#else:
    #print("Valore non esiste")

folder="Model-1"
version="1"
#epoche = readJson(folder+"/ModelTrained.json",version,"hyperparametr","epochs")
#print(epoche)
    
"""FUNZIONA"""
#writeJsonMargin("Dataset\dataSetJson.json","margineMean",24242)
"""FUNZIONA"""
#x = torch.tensor([[1, 2, 3], [4, 5, 6]])
#print(x)
#print(type(x))
#array= x.numpy()
#print(array)
#lista = array.tolist()
#value = {"margineMean":lista, "timeComputing":400.5}
#addKeyValue("Dataset\dataSetJson.json","margine",value)
#media=[2,3,4]
time="24 sec"
#obj={"margine":media,"timeComputing":time}
#addKey("Dataset\dataSetJson.json","margineMean_45",obj)

#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)

#start.record()
# whatever you are timing goes here
#end.record()

# Waits for everything to finish running
#torch.cuda.synchronize()

#print(start.elapsed_time(end))  # milliseconds
          
#print(start.elapsed_time(end))  # milliseconds

#start = timer()
# ...
#end = timer()
#print(end - start)

directory="Model-5"
array_loss_train=[2,3,4,5]
array_loss_valid=[24,453,3]
array_accuracy_train=[324,553,24]
array_accuracy_valid=[424,645,325]
array_glb_train=[434,64,324]
array_glb_valid =[3,86,456]
#saveArray(directory, "10", array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train, array_glb_valid)