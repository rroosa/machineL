# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:08:45 2020

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
from os.path import join
import numpy as np
from prova import net_save
from torch.nn import functional as F


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
        elif el >=m2: # DISSIMILI
            labels.append(1)
    return labels
    

def train_siamese(embedding_net, train_loader, valid_loader, exp_name='model_1', lr=0.01, epochs=10, momentum=0.99, margin=2, logdir='logs', modeLoss=None):
    #definiamo la contrastive loss
    if not modeLoss is None:
        if modeLoss == "norm":
            print("Loss mode Norm margin = 0.5")
            criterion = ContrastiveLossNorm()
        elif modeLoss == "double":
            print("Loss mode Norm & Double margin m1= 0.3 , m2 =0.7")
            criterion = ContrastiveLossDouble()
        elif modeLoss == "soglia":
            print("Loss mode Soglia")
            criterion = ContrastiveLoss()
        
            
    else:
        print("Loss mode margine=2")
        criterion = ContrastiveLoss()
        
    optimizer = SGD(embedding_net.parameters(), lr, momentum=momentum)
        
    
    #meters
    array_loss_train = []
    array_loss_valid = []
    array_sample_train = []
    array_sample_valid = []
    array_acc_valid = []
    array_acc_train = []
    prediction_train = []
    labels_train = []
    prediction_val = []
    labels_val = []
    
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
    global_step=0
    lossTrain = 0
    lossValid = 0
    timer = Timer()   
    for e in range(epochs):
        print("Epoca ",e)
        #iteriamo tra due modalità: train e test
        for mode in ['train','valid'] :
            print("Mode ",mode)   
            loss_meter.reset()
            acc_meter.reset()
            embedding_net.train() if mode == 'train' else embedding_net.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                
                for i, batch in enumerate(loader[mode]):
                    I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    #img1, img2, label12, label1, label2
                    #l'implementazione della rete siamese è banale:
                    #eseguiamo la embedding net sui due input
                    phi_i = embedding_net(I_i)#img 1
                    phi_j = embedding_net(I_j)#img2
                    print("Etichetta reale",l_ij)
                    
                    #calcoliamo la loss
                    l = criterion(phi_i, phi_j, l_ij)
                    
                    prediction_train = []
                    labels_train = []
                    prediction_val = []
                    labels_val = []
                    
                    
                    
                    d = F.pairwise_distance(phi_i.to('cpu'), phi_j.to('cpu'))
                    print(type(d))
                    print(d.size())

                    labs = l_ij.to('cpu')
                    #print(len(labs))
                    #tensor = torch.clamp( margin-d, min = 0) # sceglie il massimo  # sceglie il massimo -- se è zero allora sono dissimili
                    #print("max",type(tensor))
                    #print("size max tensor ",tensor.size())
                    #print("tentor 1", tensor)
                    if not modeLoss is None:
                        if modeLoss == "double":
                            
                            listaEtichette = modeLossDouble(0.3, 0.7 ,d)
                        elif modeLoss =="norm":
                            
                            listaEtichette = modeLossNorm(0.5 ,d)
                        elif modeLoss =="soglia":
                            listaEtichette = modeLossSoglia(0.5, d)
                    else:
                        listaEtichette = modeLossTrad(0.8 , d)
                    
                    
                    
                    if mode == 'train':
                        prediction_train=listaEtichette
                    else:
                        prediction_val = listaEtichette

                
                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    global_step += n
        
                    #print(len(labels))
                    #print(len(prediction))
                    
               
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    n = batch[0].shape[0] #numero di elementi nel batch
                    
                    valore =l.item()
                    loss_meter.add(valore,n)
                    print(valore,n)
                    
                    if mode=='train':
                        labels_train = labs.numpy()
                        print("Lunghezza predette TRAIN ", len(prediction_train))
                        print("Lunghezza vere TRAIN ", len(labels_train))
                        acc = accuracy_score(np.array(labels_train),np.array(prediction_train))
                        acc_meter.add(acc,n)
                        
                        
                    else:
                        labels_val = labs.numpy()
                        print("Lunghezza predette VALID ", len(prediction_val))
                        print("Lunghezza vere VALID ", len(labels_val))
                        acc = accuracy_score(np.array(labels_val),np.array(prediction_val))
                        acc_meter.add(acc,n)
                    
                    if mode=='train':
                        l_m_v = loss_meter.value()
                        print(l_m_v)
                        writer.add_scalar('loss/train', l_m_v, global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                    
          
                        
                    if mode == 'train':
                        #lossTrain =  loss_meter.value()
                        global_step_train = global_step
                        array_loss_train.append(l_m_v)
                        array_acc_train.append(acc_meter.value())
                        array_sample_train.append(global_step_train)
                        print("TRAIN- Epoca",e)
                        print("GLOBAL STEP TRAIN",global_step_train )
                        print("LOSS TRAIN", l_m_v)
                        print("ACC TRAIN", acc_meter.value())
                
                    else:
                        
                        lossValid = loss_meter.value()
                        global_step_val = global_step
                        array_loss_valid.append(lossValid)
                        array_acc_valid.append(acc_meter.value())
                        array_sample_valid.append(global_step_val)
                        print("VALID- Epoca",e)
                        print("GLOBAL STEP VALID",global_step_val )
                        print("LOSS VALID", lossValid)
                        print("ACC VALID", acc_meter.value())
                        
            writer.add_scalar('loss/'+mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar('accuracy/'+ mode, acc_meter.value(), global_step=global_step)
               
                
                
            
        #aggiungiamo un embedding. Tensorboard farà il resto
        #Per monitorare lo stato di training della rete in termini qualitativi, alla fine di ogni epoca stamperemo l'embedding dell'ultimo batch di test.
        writer.add_embedding(phi_i, batch[3], I_i, global_step=global_step, tag=exp_name+'_embedding')
        #conserviamo solo l'ultimo modello sovrascrivendo i vecchi
        
        #torch.save(embedding_net.state_dict(),'%s.pth'%exp_name) # salvare i parametri del modello

        net_save(epochs, embedding_net.state_dict(), optimizer, lossTrain ,lossValid,array_acc_train[-1],array_acc_valid[-1], global_step_train, global_step_val,'%s.pth'%exp_name)
    f='{:.7f}'.format(timer.stop())
            
    return embedding_net, f, array_loss_train, array_loss_valid, array_sample_train, array_sample_valid,  array_acc_train, array_acc_valid,labels_train,prediction_train,labels_val,prediction_val