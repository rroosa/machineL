# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:29:51 2020

@author: rosaz
"""
import argparse
import sys
import errno
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from numpy import array

import torch
import jsonschema
from torch.nn import functional as F

def writeJsonNorma(path,media,dev,time):
    """serve"""
    #media = media.tolist()
    #dev = dev.tolist()
    if not os.path.exists(path):
        entry={"normalize":{"mean":media,"dev_std":dev,"time":time}}
        with open(path, "w") as outfile:
            json.dump(entry,outfile,indent=2)
    else:
        with open(path, "r") as outfile:
            data=json.load(outfile)
        
        if not (data.get("normalize") is None):
            #print("value is present for given JSON key")
            #print(data.get("normalize"))
            #aggiungi chiavi
            entry ={"media":media,"dev":dev,"computeTime":time}
            data["normalize"]=entry
            with open(path, "w") as outfile:
                json.dump(data,outfile,indent=2)
        else:
            entry ={"media":media,"dev":dev,"computeTime":time}
            data["normalize"]=entry

            with open(path, "w") as outfile:
                json.dump(data,outfile,indent=2)
        
"""          
    if not (data.get(dev) is None):
        #print("value is present for given JSON key")
        print(data.get(dev))
        #aggiungi chiavi
        data["dev_std"]=dev
        with open(path, "w") as outfile:
            json.dump(data,outfile,indent=2)

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
 """   
def controlFile2():
   try:
    with open("Dataset/train.csv") as f:
        print("ok")
    with open("Dataset/valid.csv") as f:
        print("ok")
    with open("Dataset/test.csv") as f:
        print("ok")
        # File exists
   except IOError as e:
    print("fatal error", file=sys.stderr)
    exit()
    # Raise the exception if it is not ENOENT (No such file or directory)
    if e.errno != errno.ENOENT:
        print("fatal error", file=sys.stderr)
        exit(0)
        
        print("ciao")

def createFolder(path):
    access_rights = 0o777
    try:
            if not os.path.exists(path):
                os.mkdir(path,access_rights)
    except OSError:
            print("Creation of the directory %s failed" % path)
    else:
            print("exist the directory %s" % path)



def controlFile(path):
   try:
    with open(path+"\\normalize.json") as f:
        response = input("Do you want to re-calculate the mean and standard deviation? y | n : ")
        if(response =="y"):
            print("recalculate")
            
        elif (response =="n"):
            print("no")
        else:
            controlFile()
                     
   except IOError as e:
    print("Normalize")
    
    # Raise the exception if it is not ENOENT (No such file or directory)
    if e.errno != errno.ENOENT:
        print("fatal error", file=sys.stderr)
        exit(0)
        
def readNorm(path):
    with open(path+'\\normalize.json') as json_file:
        data = json.load(json_file)
        arrayMean = data["mean"]
        arrayDev = data["dev_std"]
        arrayMean = tuple(arrayMean)
        arrayDev = tuple(arrayDev)
        return arrayMean , arrayDev
    
    """""fuzione che aggiunge nuove keys se non esistono, mentre aggiorna valori se le chiavi esistono """
    
def controlNormalize(path):
        #controlla se è presente la directory, altrimenti la crea 
        createFolder(path)
        #print("Controll")
        if not os.path.exists(path+'\\dataSetJson.json'):
            print("1) Checking: mean, dev_std")
            
        else: # se il file esiste controlla se ci sono le key mean e dev 
            
            try:
                with open(path+"\\dataSetJson.json","r") as json_file:
                    data = json.load(json_file)
                print(path+"\\dataSetJson.json")
                if not (data.get('normalize') is None):
                    
                    norm = data['normalize']
                    print(norm)
                    if not  (norm.get('mean') and norm.get('dev_std')) is None:
                        
                        response = input("Do you want to re-calculate the mean and standard deviation? y | n : ")
                        if(response =="y"):
                            print("recalculate")
                            
                        elif (response =="n"):
                            print("bypass this step!!")
                            media = tuple(norm['mean'])
                            print(media)
                            dev= tuple(norm['dev_std'])
                            print(dev)
                            
                            
                        else:
                            controlNormalize()
                    else:
                        print("non esiste mean e dev_std, ricalcola") 
                        
                else:
                    print("non esiste normalize")
            except:
                # se il parsing è errato ricalcola la media e dev
                 print("Il parsing è errato")    
    

def writeJsonAccuracy(path, fileName, entry, accuracy, entryTime, time):
    #a_cc = {entry: accuracy}
    #timeTrain = {entryTime: time}
    
    # se il file non esistw crealo nuovo e scrivi l'oggetto 
    createFolder(path)
    if not os.path.exists(path+"\\"+fileName):
         print("File non esiste")
         entry = {entry: accuracy, entryTime: time}
         with open(path+"\\"+fileName,"w") as outfile:
             json.dump(entry,outfile) 
             
 
    #altrimenti se il file esiste 
    #prova a fare il parsing
    else:
        print("Qui")
        try:
    # Read in the JSON document,  parsing è stato effettuato con successo
            with open(path+"\\"+fileName,"r") as outfile:
                print("qui3")
                datum = json.load(outfile)
                
                # modifica il valore della chiave se esiste
            if not (datum.get(entry) is None):
                print("value is present for given JSON key")
                print(datum.get(entry))
                datum[entry]=accuracy
                with open(path+"\\"+fileName, "w") as outfile:
                    json.dump(datum, outfile)
            else:
                print("Chiave non esiste")
                    #entry = {entry: accuracy, entryTime: time}
                datum[entry]=accuracy
                with open(path+"\\"+fileName, "w") as json_outfile:
                    json.dump(datum, json_outfile)
                    
            if not (datum.get(entryTime) is None):
                print("value is present for given JSON key")
                print(datum.get(entryTime))
                datum[entryTime]=time
                with open(path+"\\"+fileName, "w") as json_outfile:
                    json.dump(datum, json_outfile)
            else:
                print("Chiave non esiste")
                
                datum[entryTime]=time
                with open(path+"\\"+fileName, "w") as json_outfile:
                    json.dump(datum,json_outfile)
                
        except:
            print("Qui2")
            entry = {entry: accuracy, entryTime: time}
            with open(path+"\\"+fileName, "w") as outfile:
                 json.dump(entry,outfile)
    
    
    
def plot(path="Model-1"):
    plt.figure()
    plt.subplot(121)
    plt.ylabel('loss train')
    plt.xlabel('num samples')
    plt.grid()
    plt.plot( [1, 2, 3, 4], [1, 4, 9, 16])
        
    plt.subplot(122)
    plt.plot([1, 2, 3, 3,2,4], [1,5,6, 4, 9, 16])
    plt.ylabel('loss validation')
    plt.xlabel('num samples')
    plt.grid()
     
    plt.savefig(path+'\\filename.png', dpi = 600)
    plt.show()

def writeJsonAppend(path, num, accuracy):
    
    entry = {'acc': accuracy, 'time': "wdd"}
    
            
    a = []
    if not os.path.isfile(path+"\\nuovo.json"):
        a.append(entry)
        with open(path+"\\nuovo.json", mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(path+"\\nuovo.json") as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(entry)
        with open(path+"\\nuovo.json", mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

def writeJsonUpdate(path, num, accuracy):
    
    entry = {'acc': accuracy, 'time': "wdd"}
            
    a = []
    if not os.path.isfile(path+"\\nuovo.json"):
        a.append(entry)
        with open(path+"\\nuovo.json", mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(path+"\\nuovo.json") as feedsjson:
            feeds = json.load(feedsjson)
            if feeds["accuracy"]:
                feeds["acc"]=2
                f.write(json.dumps(feeds, indent=2))
            
        with open(path+"\\nuovo.json", mode='w') as f:
            f.write(json.dumps(feeds, indent=2))
            
def arrayLogic():
    x = np.array([4, 3,3,3,3, 2, 1])
    print(x)
    print(type(x))
    print(len(x))
    y=[]
    for el in x:
        if el==3:
            y.append(1)
        else:
            y.append(0)
    print(y)
    """
    print(y)
    print(type(y))
    print(len(y))"""


def distanza():
    A = torch.Tensor([
        [[1,2,3],    [4,5,6],    [7,8,9]],
        [[11,12,13], [14,15,16], [17,18,19]],
        [[21,22,23], [24,25,26], [27,28,29]],
        ])
    print(A)
    print(A.size())
    margin = 2
    margin2 = 1

    B = torch.Tensor([
        [[1,2,3],    [4,5,6],    [7,8,9]],
        [[11,12,13], [14,15,16], [17,18,19]],
        [[21,22,23], [24,25,26], [27,28,29]],
        ])
    
    C = A*4

    d = F.pairwise_distance(A, B)
    
    print("di",d)
    
    print("Margin-di",margin-d)
    tensor = torch.clamp( margin-d, min = 0) # sceglie il massimo -- se è zero allora sono dissimili
    print("max m-d",tensor)
    tensorSimil= torch.Tensor([0])
    tensorDissimil= torch.Tensor([1])
    result= torch.where(tensor==0.,tensorDissimil, tensorSimil)
    print("max result Label", result)
    print(result[0][0])
    if(result[0][0]==1.):
        label= 1
        print("Dissimili",label)
    else:
        label = 0
        print("Simil",label)
        
        
        
    di = F.pairwise_distance(A, C)
    
    print("di",di)
    
    print("Margin-di",margin-di)
    tensor = torch.clamp( margin-di, min = 0) # sceglie il massimo -- se è zero allora sono dissimili
    print("max m-d",tensor)
    tensorSimil= torch.Tensor([0])
    tensorDissimil= torch.Tensor([1])
    result= torch.where(tensor==0.,tensorDissimil, tensorSimil)
    print("max result Label", result)
    print(result[0][0])
    if(result[0][0]==1.):
        label= 1
        print("Dissimili",label)
    else:
        label = 0
        print("Simil",label)
        
    
    #matrix = tensor.numpy()
    #print("Matrix",matrix.ravel(), type(matrix))
    #list(matrix)
    #print(np.all([n<=margin for n in tensor]))
    """
    if(tensor <= margin):
        print("Simili A e B")
    else:
        print("Dissimili A e B")
    """

def readFileDataset(path,entry):
    if not os.path.exists(path):
        print("Dataset is not present, try --create dataset", file=sys.stderr)
        exit(0)
    else:
        with open(path, "r") as outfile:
            data= json.load(outfile)
        if not (data.get(entry) is None):
            value = data[entry]
            return value
            
def lengthDataset(path,entry,key):
    somma = 0
    if not os.path.exists(path):
        print("Dataset is not present, try --create dataset", file=sys.stderr)
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

def print_to_stderr(*a): 
  
    # Here a is the array holding the objects 
    # passed as the arguement of the function 
    print(*a, file = sys.stderr) 
  
## AGGIIUNGI FUNZIONA 
def addJsonModel(directory,version, acc ,f1score, precision, recall, time):
    path = "provaJson.json"
    
    if not os.path.exists(path):
        print("File %s not is exists" % path)
        sys.stderr.write("File %s not is exists" % path)
        exit(0)
    else:
        #leggi il file
        with open(path, "r") as outfile:
            data = json.load(outfile)
        if not (data.get(version) is None): # se la versione esiste gia, aggiorna i campi
            print(data.get(version))
            versione = data[version]
            if not (versione.get("accuracy") is None): # se la chiave accuracy esiste, aggiorna i campi
                obj = versione["accuracy"]
                if not (obj.get("accuracyTets") is None):
                    obj["accuracyTest"]=acc
                    with open(path, "w") as outfile:
                        json.dump(data,outfile,indent=2)
                else:
                    obj["accuracyTest"]=acc
                    with open(path, "w") as outfile:
                        json.dump(data,outfile,indent=2)
            else:
                print("non esiste")
                value={"accuracyTest":acc}
                versione["accuracy"]=value

                with open(path, "w") as outfile:
                    json.dump(data,outfile,indent=2)

                                #accuracy, accuracyTest ,joj
def addValueJsonModel(path,num, key ,entry, value):
    path = "provaJson.json"
    
    if not os.path.exists(path):
        print("File %s not is exists" % path)
        sys.stderr.write("File %s not is exists" % path)
        exit(0)
    else:
        #leggi il file
        with open(path, "r") as outfile:
            data = json.load(outfile)
        if not (data.get(num) is None): # se la versione esiste gia, aggiorna i campi
            print(data.get(num))
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
                print("non esiste")
                obj={entry:value}
                versione[entry]=obj

                with open(path, "w") as outfile:
                    json.dump(data,outfile,indent=2)
        else:
            dato={key:{entry:value}}
            data[num]=dato
            with open(path, "w") as outfile:
                json.dump(data,outfile,indent=2)
            
    
        

    
def writeJson(model,num, media, dev, time):
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
"""        
a = np.arange(10).reshape(2,5) # a 2 by 5 array
b = a.tolist()
writeJson("Model-1",b,1243,33,"2e33333sec")      
"""
#controlFile("Model-1")  
#arrayM, arrayD = readNorm("Model-1")
#print(arrayM)
#print(arrayD)

#plot("Model-1")
entry= "acc"
entryTime="nuvissima"
time="3haf"
accuracy =125
path="Model-1"
#writeJsonAccuracy(path,"normalize.json",entry, accuracy, entryTime, time)
media=[2,4,3,4]
dev=[3,4,5,4]
time="23sec"
#writeJsonNorma("Dataset\dataSetJson.json",media,dev,time)
"""
value=readFileDataset("Dataset\dataSetJson.json", "datasetLarge")
for i in value:
    print(i)
    
key = "num_sample"
num = lengthDataset("Dataset\dataSetJson.json","datasetLarge",key)
print(num)
#ok distanza()

#
#arrayLogic()
"""
"""
parser = argparse.ArgumentParser( description = "Dataset Money")
    
parser.add_argument('--model', help = "Name of model [modello1 | modello2 | modello3]", type=str)
parser.add_argument('--v', help ="version" , type=int)

args = parser.parse_args()

required_together = ('model','v')

# args.b will be None if b is not provided
if not all([getattr(args,x) for x in required_together]):
    raise RuntimeError("Cannot supply --model without --v")
else:
    if args.model == "model1":
        print("model ",args.model)
        if args.v == 1:
            print("version",args.v)
        else:
            print("Versione non riconoscita [1 | 2 | 3]")
            print_to_stderr("Hello World")
    else:
        print("Modello non riconosciuto [modello1 | modello2 | modello3]")
        
        print(type(sys.stderr))
"""
#sys.stderr.write("Error messages can go here\n")
acc="uffa"
f1score="score111"
precision="perfect"
recall="recallll"
time="142sec"

#addJsonModel("provaJson.json","1", acc ,f1score, precision, recall, time)

key="accuracy"
entry="accuracyTrain"
value="ValoreAcc"
#addValueJsonModel("provaJson.json","1", key ,entry, value)
key="time"
entry="timeTrain"
value="ValoreTime"
#addValueJsonModel("provaJson.json","1", key ,entry, value)

controlNormalize("Dataset")