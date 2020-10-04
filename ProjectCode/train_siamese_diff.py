# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:15:35 2020

@author: rosaz
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from ContrastiveLoss import ContrastiveLoss
from ContrastiveLossNorm import ContrastiveLossNorm
from ContrastiveLossDouble import ContrastiveLossDouble
from AverageValueMeter import AverageValueMeter
from utils.calculate_time import Timer
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torchvision import transforms
from timeit import default_timer as timer
from os.path import join
import numpy as np
from prova import net_save, writeJsonModelEpoca, saveArray
from torch.nn import functional as F
from matplotlib import pyplot as plt
from scipy.stats import norm
import math
import statistics

def modeLossNorm(m, d ):
    labels = []
    
    d_n = 2.0*( 1.0/(1.0 + torch.exp(-1*d)) - 0.5)
    print("distanze normalizzate", d_n)
    
    for el in d_n:
        if el <= 0.5: # SIMILI
            
            labels.append(0)
        else: # DISSIMILI
            
            labels.append(1)
    return labels

def modeLossTrad(m, d ):
    labels = []
    print("distanze ", d)
    for el in d:
       if el <= m: # SIMILI
            labels.append(0)
       else: # DISSIMILI
           labels.append(1)
           
    return labels

def modeLossSoglia(m, d ):
    labels = []
    
    
    print("distanze con soglia", d)
    
    for el in d:
        if el <= 0.5: # SIMILI
            
            labels.append(0)
        else: # DISSIMILI
            
            labels.append(1)
    return labels
    

def modeLossDouble(m1,m2, d):
    labels = []
    
    d_n = 2.0*( 1.0/(1.0 + torch.exp(-1*d)) - 0.5)
    print("distanze normalizzate double", d_n)
    
    for el in d_n:
        if el <= m1: # SIMILI
            
            labels.append(0)
        elif el >= m2: # DISSIMILI
            labels.append(1)
    return labels
    

def train_siamese_diff(directory, version, embedding_net,train_loader, valid_loader,resize, batch_size,margin1,margin2, exp_name='model_1', lr=0.01, epochs=10, momentum=0.99,  logdir='logs',decay=None, modeLoss=None, dizionario_array=None):
    #definiamo la contrastive loss
    print("lr",lr)
    print("momentum",momentum)
    print("decay",decay)
    print("margin1", margin1)
    print("margine2",margin2)
    
    if not modeLoss is None:
        if modeLoss == "norm":
            print("Loss mode Norm margin = 0.5")
            criterion = ContrastiveLossNorm()
        elif modeLoss == "double":
            print("Loss mode Double margin m1= 0.3 , m2 =0.7")
            criterion = ContrastiveLossDouble()
        elif modeLoss == "soglia":
            print("Loss mode Soglia")
            criterion = ContrastiveLoss()
            
        elif modeLoss == "due":
            print("Loss mode due")
            criterion = ContrastiveLossDouble(margin1, margin2)
              
    else:
        print("Loss mode margine=2")
        criterion = ContrastiveLoss()
        
    if not decay is None:                                         
        print("Weight_Decay",decay)
        optimizer = SGD(embedding_net.parameters(), lr, momentum=momentum,weight_decay=decay)
    
    else:        
        optimizer = SGD(embedding_net.parameters(), lr, momentum=momentum)       
    
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
     

    #writer
    writer = SummaryWriter(join(logdir, exp_name))
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    embedding_net.to(device)
    criterion.to(device)# anche la loss va portata sul device in quanto contiene un parametro(m)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'valid' : valid_loader
    }
    last_loss_train = 0
    last_loss_val = 0
    last_acc_train = 0
    last_acc_val = 0
    
    array_accuracy_train = []
    array_accuracy_valid = []    
    array_loss_train = []
    array_loss_valid = []
    array_glb_train = []
    array_glb_valid = []
    tempo = Timer()
    global_step=0
    start_epoca = 0    
    
    if dizionario_array is not None:
        print("Inizializza")
        array_accuracy_train = dizionario_array["a_train"]
        array_accuracy_valid = dizionario_array["a_valid"]    
        array_loss_train = dizionario_array["l_train"]
        array_loss_valid = dizionario_array["l_valid"]
        array_glb_train = dizionario_array["g_train"]
        array_glb_valid = dizionario_array["g_valid"]
        global_step= dizionario_array["g_valid"][-1]
        start_epoca = dizionario_array["epoche_fatte"] + 1 # indice epoca di inizio
    
    print("global step", global_step)
    print("a_acc_train", array_accuracy_train)
    print("a_acc_valid",array_accuracy_valid)
    print("loss_train", array_loss_train)
    print("loss_valid",array_loss_valid)
    print("glb_train",array_glb_train)
    print("glb_valid",array_glb_valid)
    print("epoca_start_indice ", start_epoca)    
    

    start = timer() 
    
    print("Num epoche", epochs)  
    
    for e in range(start_epoca,epochs):
        print("Epoca ",e)
        
        array_total_0 = []
        array_total_1 = []
        #iteriamo tra due modalità: train e test
        for mode in ['train','valid'] :
            print("Mode ",mode)   
            loss_meter.reset()
            acc_meter.reset()
            embedding_net.train() if mode == 'train' else embedding_net.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                
                for i, batch in enumerate(loader[mode]):
                    
                    distance_1 = []
                    distance_0 = []                    
                    
                    I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    #img1, img2, label12, label1, label2
                    #l'implementazione della rete siamese è banale:
                    #eseguiamo la embedding net sui due input
                    phi_i = embedding_net(I_i)#img 1
                    phi_j = embedding_net(I_j)#img2
                    
                    print("Output train img1", phi_i.size())
                    print("Output train img2", phi_j.size())
                    print("Etichetta reale",l_ij)
                    
                    #calcoliamo la loss
                    l = criterion(phi_i, phi_j, l_ij)
                    
                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    print("Num elemnti nel batch ",n)
                    global_step += n
                    
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()                    
                    
                    
                    dist = F.pairwise_distance(phi_i, phi_j)
                    dist = dist.detach().cpu()
                    dist = dist.tolist()
                    print("DISTANZE ",dist)
                    
                    """
                    prediction_train = []
                    labels_train = []
                    prediction_val = []
                    labels_val = []
                    """
                    """
                    d = F.pairwise_distance(phi_i.to('cpu'), phi_j.to('cpu'))
                    print(type(d))
                    print(d.size())
                    """
                    res=torch.abs(phi_i.cuda() - phi_j.cuda())
                    res=res.detach().cpu()
                    
                    #print("Tipo",type(res))
                    #print("DIm",res.size())
                    labs = l_ij.to('cpu')
                    label=[]
                    lab_batch_predette =[]
                    for  i, el  in enumerate (res):
                        print("Tipo",type(el))
                        print("DIm",el.size())
                        print("posiz 0",el[0])
                        print("posizione 1",el[1])
                        result, indice=torch.max(el,0)
                        indice= indice.item()
                        print("PREDETTA di un smaple",indice)
                        labelv = l_ij[i].to('cpu')
                        labelv= labelv.item()
                       
                        print(" REALE di un sample",labelv)
                    
                        if labelv == indice:
                            print("Corretta R - P",labelv,indice)
                            
                        
                        else:
                            print("Scorretta R - P",labelv,indice)
                        
                        lab_batch_predette.append(indice)
                    
                    label.extend(list(labs.numpy()))
                    print("Predette", lab_batch_predette)
                    print("Reali", label)
                  
                    acc = accuracy_score(np.array(label),np.array(lab_batch_predette))
                    n = batch[0].shape[0] #numero di elementi nel batch
                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)                       

              
                    
                    if mode=='train':
                        
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                       
                        distance_1 = [distance for distance , label in zip (dist, labs.numpy()) if label == 1]
                        distance_0 = [distance for distance, label in zip (dist, labs.numpy()) if label == 0]
                        if(len(distance_0)!=0):
                            array_total_0.extend(distance_0)
                        if(len(distance_1)!=0):
                            array_total_1.extend(distance_1)

            if mode == 'train':
                global_step_train=global_step
                last_loss_train = loss_meter.value()
                last_acc_train = acc_meter.value()
                
                array_accuracy_train.append(acc_meter.value())
                array_loss_train.append(loss_meter.value())
                array_glb_train.append(global_step)
            else:
                global_step_val = global_step
                last_loss_val = loss_meter.value()
                last_acc_val = acc_meter.value()
                
                array_accuracy_valid.append(acc_meter.value())
                array_loss_valid.append(loss_meter.value())
                array_glb_valid.append(global_step)
            
            writer.add_scalar('loss/'+mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar('loss/'+mode, acc_meter.value(), global_step=global_step)
               
        print("Loss TRAIN",array_loss_train)
        print("Losss VALID",array_loss_valid)
        print("Accuracy TRAIN",array_accuracy_train)
        print("Accuracy VALID",array_accuracy_valid)
        print("dim acc train",len(array_accuracy_train))
        print("dim acc valid",len(array_accuracy_valid))
        figure = plt.figure(figsize=(12,8))
        plt.plot(array_glb_train,array_accuracy_train)
        plt.plot(array_glb_valid,array_accuracy_valid)
        plt.xlabel('samples')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotAccuracy_'+version+'.png')
        plt.clf()
        plt.close(figure)
        #plt.show()
        
        figure= plt.figure(figsize=(12,8))
        plt.plot(array_glb_train,array_loss_train)
        plt.plot(array_glb_valid,array_loss_valid)
        plt.xlabel('samples')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotLoss_'+version+'.png')
        plt.clf()
        plt.close(figure)

        #plt.show()
        
        
        
        

        #aggiungiamo un embedding. Tensorboard farà il resto
        writer.add_embedding(phi_i, batch[3], I_i, global_step=global_step, tag=exp_name+'_embedding')
        #conserviamo solo l'ultimo modello sovrascrivendo i vecchi
        torch.save(embedding_net,'%s.pth'%exp_name)
        # conserviamo il odello a questa epoca
        torch.save(embedding_net,directory+"//"+version+"//"+'%s.pth'%(exp_name+"_"+str(e)))
        
        #conserviamo il modello sotto forma di dizionario
        net_save(epochs, embedding_net, optimizer, last_loss_train ,last_loss_val,last_acc_train,last_acc_val, global_step_train, global_step_val,'%s.pth'%(exp_name+"_dict"))
        #dipo ogni epoca plotto la distribuzione
        print("lungezza array_total_0 ",len(array_total_0))
        print("lunghezza array_total_1",len(array_total_1))
        
        
        saveArray(directory,version,array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train,array_glb_valid)
        
        saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,len(train_loader),array_accuracy_train[-1],array_accuracy_valid[-1], array_loss_train[-1],array_loss_valid[-1])
        
        
        draw_distribution(directory, version , e, array_total_0, array_total_1)
        
    f='{:.7f}'.format(tempo.stop())  
     
    return embedding_net, f, last_loss_train,last_loss_val, last_acc_train,last_acc_val


def saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,pair,last_acc_train,last_acc_val, last_loss_train,last_loss_val):
    end = timer()
    time = end-start
    hyperparametr = {"indexEpoch":e,"lr":lr, "momentum" : momentum, "numSampleTrain": pair }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": time}
    writeJsonModelEpoca(directory,version,hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)

def draw_distribution(directory,version,e,array_total_0,array_total_1):
    mu_0 = statistics.mean(array_total_0)
    print("Media 0:",mu_0)
    somma = 0
    for i in array_total_0:
        somma = somma + math.pow(i-mu_0,2)
    
    sigma_0 = math.sqrt(somma / len(array_total_0))
    
    print("Dev_std_0:",sigma_0)
    
    # ---------------------------
    mu_1 = statistics.mean(array_total_1)
    print("Media_1:",mu_1)
    somma = 0
    for i in array_total_1:
        somma = somma + math.pow(i-mu_1,2)
    
    sigma_1 = math.sqrt(somma / len(array_total_1))
    
    print("Dev_std_1:",sigma_1)
    
    
    
    g_0 = norm(mu_0,sigma_0)
    g_1 = norm(mu_1,sigma_1)
    x_0=np.linspace(0, max(array_total_0),100)
    x_1=np.linspace(0, max(array_total_1),100)
    plt.figure(figsize=(15,6))
    media_0='{:.3f}'.format(mu_0)
    media_1='{:.3f}'.format(mu_1)
    plt.hist(array_total_0, bins=100, density = True)
    plt.hist(array_total_1, bins=100, density = True)

    plt.plot(x_0, g_0.pdf(x_0))
    plt.plot(x_1, g_1.pdf(x_1))
    plt.grid()
    plt.title("Media_0: "+media_0+"   Media_1: "+media_1)
    plt.legend(['Densità Stimata_0','Densità Stimata_1','Distribuzione Gaussiana_0','Distribuzione Gaussiana_1'])
    plt.savefig(directory+"\\"+version+"\\"+'plotDistribution_'+str(e)+'.png')
    plt.clf()
    #plt.show()
    
    