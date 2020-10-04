# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:34:06 2020

@author: rosaz
"""


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from timeit import default_timer as timer
from AverageValueMeter import AverageValueMeter
from utils.calculate_time import Timer
from sklearn.metrics import accuracy_score
from torch import nn
import numpy as np
from torch.nn import functional as F
from prova import net_save, writeJsonModelEpoca, saveArray
from os.path import join
from matplotlib import pyplot as plt
from ContrastiveLoss import ContrastiveLoss
  
def train_siamese_distrib_margine(directory, version, model, train_loader, valid_loader,resize,batch_size, exp_name='model_1',margine, decay=None, lr=0.0001, epochs=10, momentum=0.99,   logdir='logs',  dizionario_array =None, modeLoss = None):
    print("momonetum",momentum)
    print("lr",lr)
    
    
    if not modeLoss is None:
        if modeLoss == "single":
            criterion = ContrastiveLoss(margine)
    
    if not decay is None:                                         
        print("Weight_Decay",decay)
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=decay)
    else:
        optimizer = SGD(model.parameters(), lr, momentum=momentum)
         
    if not dizionario_array is None:
        optimizer.load_state_dict(dizionario_array["optimizer"])
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #writer
    writer = SummaryWriter(join(logdir, exp_name))
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion.to(device)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'valid' : valid_loader
        }
    
    
    if not dizionario_array is None:
        array_accuracy_train = dizionario_array["a_train"]
        array_accuracy_valid = dizionario_array["a_valid"]    
        array_loss_train = dizionario_array["l_train"]
        array_loss_valid = dizionario_array["l_valid"]
        array_glb_train = dizionario_array["g_train"]
        array_glb_valid = dizionario_array["g_valid"]
        global_step = array_glb_valid[-1]
        last_loss_train = array_loss_train[-1]
        last_loss_val = array_loss_valid[-1] 
        last_acc_train = array_accuracy_train[-1] 
        last_acc_val = array_accuracy_valid[-1] 
        epoche_fatte = dizionario_array["epoche_fatte"]
        epoche_avanza=dizionario_array["epoche_avanza"]
        
    else:
        array_accuracy_train = []
        array_accuracy_valid = []    
        array_loss_train = []
        array_loss_valid = []
        array_glb_train = []
        array_glb_valid = []
        global_step = 0
        last_loss_train = 0
        last_loss_val = 0
        last_acc_train = 0
        last_acc_val = 0
    #inizializziamo il global step
    
    tempo = Timer() 
    start = timer()
    
    
    for e in range(epochs):
        print("Epoca= ",e)
        #iteriamo tra due modalità: train e test
        for mode in ['train','valid']:
            loss_meter.reset(); 
            acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                

                for i, batch in enumerate(loader[mode]):
                    print("Num batch =", i)
                    I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    #img1, img2, label12, label1, label2
                    #l'implementazione della rete siamese è banale:
                    #eseguiamo la embedding net sui due input
                    phi_i = model(I_i)#img 1
                    phi_j = model(I_j)#img2
                    
                    print("Output train img1", phi_i.size())
                    print("Output train img2", phi_j.size())
                    #print("Etichetta reale",l_ij)
                    euclidean_distance = F.pairwise_distance(phi_i, phi_j)  
                    
                    euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # distanza
                    labs = l_ij.to('cpu').numpy() # etichette reali
                    
                    etichette_predette = [ euclid_tmp > margine]
                    print(etichette_predette)
                    etichette_predette = etichette_predette.int()
                    
                    
                    #l_ij = l_ij.type(torch.LongTensor).to(device)
                    #calcoliamo la loss
                    l = criterion(phi_i, phi_j , l_ij)
                    
                    

                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    #print("numero elementi nel batch ",n)
                    global_step += n
                    
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    acc = accuracy_score(np.array(labs),np.array(etichette_predette.numpy()))
                    n = batch[0].shape[0]
                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)
                    #loggiamo i risultati iterazione per iterazione solo durante il training
                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                    #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
            
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
               
            writer.add_scalar('loss/' + mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar('accuracy/' + mode, acc_meter.value(), global_step=global_step)
            
        
        print("Loss TRAIN",array_loss_train)
        print("Losss VALID",array_loss_valid)
        print("Accuracy TRAIN",array_accuracy_train)
        print("Accuracy VALID",array_accuracy_valid)
        print("dim acc train",len(array_accuracy_train))
        print("dim acc valid",len(array_accuracy_valid))
        plt.figure(figsize=(12,8))
        plt.plot(array_glb_train,array_accuracy_train)
        plt.plot(array_glb_valid,array_accuracy_valid)
        plt.xlabel('samples')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotAccuracy_'+version+'.png')
        plt.show()
        
        plt.figure(figsize=(12,8))
        plt.plot(array_glb_train,array_loss_train)
        plt.plot(array_glb_valid,array_loss_valid)
        plt.xlabel('samples')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotLoss_'+version+'.png')
        plt.show()
        
        saveArray(directory,version,array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train,array_glb_valid)
        
        saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,len(train_loader),array_accuracy_train[-1],array_accuracy_valid[-1], array_loss_train[-1],array_loss_valid[-1])
        
        
        #writer.add_embedding(phi_i, batch[3], I_i, global_step=global_step, tag=exp_name+'_embedding')
        #conserviamo i pesi del modello alla fine di un ciclo di training e test
        net_save(epochs, model, optimizer, last_loss_train ,last_loss_val,last_acc_train,last_acc_val, global_step_train,global_step_val,'%s.pth'%(exp_name+"_dict"))
        torch.save(model,'%s.pth'%exp_name)
        torch.save(model,directory+"//"+version+"//"+'%s.pth'%(exp_name+"_"+str(e)))
    f='{:.7f}'.format(tempo.stop())
    return model ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val


def saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,pair,last_acc_train,last_acc_val, last_loss_train,last_loss_val):
    end = timer()
    time = end-start
    hyperparametr = {"epochs":e,"lr":lr, "momentum" : momentum, "numSampleTrain": pair }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": time}
    writeJsonModelEpoca(directory,version,hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
 

