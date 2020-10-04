# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:53:57 2020

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torchvision import transforms
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
from prova import net_save, writeJsonModelEpoca, saveArray, saveArray_metod2
from torch.nn import functional as F
from timeit import default_timer as timer
from scipy.stats import norm
import math
import statistics

def train_siamese_margin_double(directory,version,embedding_net, train_loader, valid_loader,resize,batch_size, exp_name='model',lr=0.01, epochs=10, momentum=0.99, margin1=0.8,margin2=1.3, logdir='logs', decay=None, modeLoss=None, dizionario= None):
#definiamo la contrastive loss
    print("lr",lr)
    print("momentum",momentum)
    print("decay",decay)
    print("margin1", margin1)
    print("margin2",margin2)
    
    #definiamo la contrastive loss
    if not modeLoss is None:
        if modeLoss == "double":
            print("Loss mode Double margin ")
            criterion = ContrastiveLossDouble(margin1, margin2)
        
        
    
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
    accuracy_metodo2_validation= []
    
    array_f1_valid =[]
    array_recall_valid = [] 
    array_precision_valid = [] 
    tp_valid = []
    fp_valid = [] 
    
    
    
    array_loss_train = []
    array_loss_valid = []
    array_glb_train = []
    array_glb_valid = []
    tempo = Timer()
    global_step=0
    start_epoca = 0
    tempoTrain = 0
    
    if dizionario is not None:
        print("Inizializza")
        array_accuracy_train = dizionario["a_train"]
        array_accuracy_valid = dizionario["a_valid"] 
        accuracy_metodo2_validation = dizionario["array_acc_valid_2"] 
        array_loss_train = dizionario["l_train"]
        array_loss_valid = dizionario["l_valid"]
        array_glb_train = dizionario["g_train"]
        array_glb_valid = dizionario["g_valid"]
        global_step= dizionario["g_valid"][-1]
        start_epoca = dizionario["epoche_fatte"] + 1 # indice epoca di inizio
        tempoTrain = dizionario["tempoTrain"]
        array_f1_valid =dizionario["array_f1_valid_2"]
        array_recall_valid = dizionario ["array_recall_valid_2"] 
        array_precision_valid =dizionario["array_precision_valid_2"] 
        tp_valid = dizionario["array_tp_valid_2"]
        fp_valid =dizionario["array_fp_valid_2"]
    
    print("global step", global_step)
    print("a_acc_train", array_accuracy_train)
    print("a_acc_valid",array_accuracy_valid)
    print("loss_train", array_loss_train)
    print("loss_valid",array_loss_valid)
    print("glb_train",array_glb_train)
    print("glb_valid",array_glb_valid)
    print("%d, %d, %d, %d, %d, %d" %(len(array_accuracy_train) , len(array_accuracy_valid),len(array_loss_train), len(array_loss_valid),len(array_glb_train),len(array_glb_valid) ))
    print("epoca_start_indice ", start_epoca)

    
    
    start = timer()  + tempoTrain
    
    print("Num epoche", epochs)
    
    for e in range(start_epoca,epochs):
        print("Epoca= ",e)

        array_total_0 = []
        array_total_1 = []

        distanze_validation = []
        label_reali_validation = []

        #iteriamo tra due modalità: train e test
        for mode in ['train','valid'] :
            loss_meter.reset()
            acc_meter.reset()
            embedding_net.train() if mode == 'train' else embedding_net.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                for i, batch in enumerate(loader[mode]):
                    distance_1 = []
                    distance_0 = []
                    

                    
                    I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    #l'implementazione della rete siamese è banale:
                    #eseguiamo la embedding net sui due input
                    phi_i = embedding_net(I_i)
                    phi_j = embedding_net(I_j)
                    
                    #print("Output train img1", phi_i.size())
                    #print("Output train img2", phi_j.size())

                    #calcoliamo la loss
                    l = criterion(phi_i, phi_j, l_ij)

                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    #print("Num elemnti nel batch ",n)
                    global_step += n
                    
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    dist = F.pairwise_distance(phi_i, phi_j)
                    dist = dist.detach().cpu()
                    dist = dist.tolist()
                    #print("DISTANZE ",dist)
                    
                    pred = []
                    label = []
                    labs = l_ij.to('cpu')
                    print("epoca %d, mode %s" %(e, mode) )
                    
                    
                    if mode =='valid':
                        distanze_validation.extend(dist)
                        label_reali_validation.extend(list(labs.numpy()))
                    
                    

                    
                    
                    for j in dist:
                        #print(j)
                        if j<= margin1:
                            #print("é minore %0.5f"%(j<= margin1))
                            pred.append(0)
                                
                        elif j>=margin2:
                            #print("E' maggiore %0.5f"%(j>=margin2))
                            pred.append(1)
                        
                        else:
                            if(abs(j - margin1) <= abs(j-margin2)):
                                #print("intervallo classe 0 :%0.5f , %0.5f"%(abs(j - margin1),abs(j-margin2)))
                                pred.append(0)
                            else:
                                #print("intervallo classe 1 :%0.5f , %0.5f"%(abs(j - margin1),abs(j-margin2)))
                                pred.append(1)
                        
                    label.extend(list(labs.numpy()))
                    
                    
                    #print("Predette", pred)
                    #print("Reali", labs)
                        
                    
                    acc = accuracy_score(np.array(label),np.array(pred))
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
        
        saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,decay,len(train_loader),array_accuracy_train[-1],array_accuracy_valid[-1], array_loss_train[-1],array_loss_valid[-1],margin1, margin2)

        draw_distribution(directory, version , e, array_total_0, array_total_1)
        
        accuracy_metodo2, f1,recall,precision, tp, fp = calculate_pred_label_metod2(directory, version , e, array_total_0, array_total_1, array_glb_valid,label_reali_validation, distanze_validation, accuracy_metodo2_validation)
        array_f1_valid.append(f1)
        array_recall_valid.append(recall)
        array_precision_valid.append(precision)
        fp_valid.append(fp)
        tp_valid.append(tp)
        accuracy_metodo2_validation= accuracy_metodo2
        print("accuracy_metodo2_validation: ",len(accuracy_metodo2_validation))
        saveArray_metod2(directory,version,accuracy_metodo2_validation, array_f1_valid , array_recall_valid, array_precision_valid,tp_valid,fp_valid, array_glb_valid)
        
        
        #print("Loss TRAIN",array_loss_train)
        #print("Losss VALID",array_loss_valid)
        #print("Accuracy TRAIN",array_accuracy_train)
        #print("Accuracy VALID",array_accuracy_valid)
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

        
    f = (tempo.stop()) + tempoTrain 
    
    return embedding_net, f, last_loss_train,last_loss_val, last_acc_train,last_acc_val



def saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,decay,pair,last_acc_train,last_acc_val, last_loss_train,last_loss_val, margin1, margin2):
    end = timer()
    time = end-start
    
    if not decay is None:
           
        hyperparametr = {"indexEpoch":e,"lr":lr,"decay":decay, "momentum" : momentum, "numSampleTrain": pair,"margin1":margin1, "margin2":margin2 }
    
        
    else:  
        hyperparametr = {"indexEpoch":e,"lr":lr, "momentum" : momentum, "numSampleTrain": pair,"margin1":margin1, "margin2":margin2 }
    
    
    
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": time}
    writeJsonModelEpoca(directory,version,hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
    
def draw_distribution(directory,version,e,array_total_0,array_total_1):
    mu_0 = statistics.mean(array_total_0)
    #print("Media 0:",mu_0)
    somma = 0
    for i in array_total_0:
        somma = somma + math.pow(i-mu_0,2)
    
    sigma_0 = math.sqrt(somma / len(array_total_0))
    
    #print("Dev_std_0:",sigma_0)
    
    # ---------------------------
    mu_1 = statistics.mean(array_total_1)
    #print("Media_1:",mu_1)
    somma = 0
    for i in array_total_1:
        somma = somma + math.pow(i-mu_1,2)
    
    sigma_1 = math.sqrt(somma / len(array_total_1))
    
    #print("Dev_std_1:",sigma_1)
    
    
    
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
    
def calculate_pred_label_metod2(directory, version , e, array_total_0, array_total_1,array_glb_valid, label_reali_validation, distanze_validation, accuracy_metodo2_validation):
    

    #la notazione sotto è una notazione compatta
    #per definire una distribuzione normale e fittarla sui dati

    #costruiamo due distribuzioni normali separate per i due set di dati
    g_1 = norm(*norm.fit(array_total_1))
    g_0 = norm(*norm.fit(array_total_0))
    
    minimo = min(g_1.ppf(0.001),g_0.ppf(0.001))
    massimo = max(g_1.ppf(0.999),g_0.ppf(0.999))
    #Plottiamo dunque le due Gaussiane considerando un range di valori  𝑥  comune:
    x = np.linspace(minimo, massimo,1000)
    
    plt.figure(figsize=(12,6))
    plt.plot(x,g_0.pdf(x))
    plt.plot(x,g_1.pdf(x))
    plt.legend(['Simil_0','Dissimil_1'])
    plt.grid()
    plt.savefig(directory+"\\"+version+"\\"+"Metod2"+"\\"+'plotPairTrain_'+str(e)+'.png')
    plt.clf()
    
    #Calcoliamo dunque le probabilità  𝑃(D | S)  per tutti i valori del valid set:
    #plt.show()
    prob_0 = g_0.pdf(distanze_validation)
    prob_1 = g_1.pdf(distanze_validation)
    
    #A questo punto costruiamo un vettore predette che contenga True se pensiamo che il soggetto sia un Dissimile, e dunque se prob_1 è maggiore di prob_0:
    
    predette = prob_1>prob_0
    
    accV = accuracy_score(np.array(label_reali_validation),np.array(predette))
    recall = recall_score(np.array(label_reali_validation),np.array(predette))
    precision = precision_score(np.array(label_reali_validation),np.array(predette))
    f1 = f1_score(np.array(label_reali_validation),np.array(predette))
    
    accuracy_metodo2_validation.append(accV)
    
    cm = confusion_matrix(np.array(label_reali_validation),np.array(predette))
    cm=cm/cm.sum(1).reshape(-1,1)
    _, fpr, _, tpr = cm.ravel()
    
    
    #print("False Positive Rate: {:0.2f}".format(fpr))
    #print("True Positive Rate: {:0.2f}".format(tpr))
    #print("Accuracy: {:0.2f}".format(accV))
    
    figure = plt.figure(figsize=(12,8))
    
    plt.plot(accuracy_metodo2_validation)
    plt.xlabel('samples')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['Valid'])
    plt.savefig(directory+"\\"+version+"\\"+"Metod2"+"\\"+'plotAccuracy_Valid.png')
    plt.clf()
    plt.close(figure)
    
      
    return accuracy_metodo2_validation, f1, recall, precision,tpr,fpr
    
    
    
    
    
