# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:12:36 2020

@author: rosaz
"""


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from AverageValueMeter import AverageValueMeter
from utils.calculate_time import Timer
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from torch import nn
from prova import net_save, saveArray
from os.path import join
from matplotlib import pyplot as plt
from prova import net_save, writeJsonModelEpoca

def train_siamese_class_adam(directory,version,model, train_loader, valid_loader,resize, batch_size, exp_name='model_1', lr=0.0001, epochs=10, momentum=0.99, margin=2, logdir='logs', modeLoss=None,dizionario =None):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr, betas = (0.9 , 0.999), weight_decay=0.0004)
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
    
    start_epoca =0
    #inizializziamo il global step
    global_step = 0
    tempo = Timer()  
    start = timer()
    
    if dizionario is not None:
        print("Inizializza")
        array_accuracy_train = dizionario["a_train"]
        array_accuracy_valid = dizionario["a_valid"]    
        array_loss_train = dizionario["l_train"]
        array_loss_valid = dizionario["l_valid"]
        array_glb_train = dizionario["g_train"]
        array_glb_valid = dizionario["g_valid"]
        global_step= dizionario["g_valid"][-1]
        start_epoca = dizionario["epoche_fatte"] + 1 # indice epoca di inizio

    print("global step", global_step)
    print("a_acc_train", array_accuracy_train)
    print("a_acc_valid",array_accuracy_valid)
    print("loss_train", array_loss_train)
    print("loss_valid",array_loss_valid)
    print("glb_train",array_glb_train)
    print("glb_valid",array_glb_valid)
    print("epoca_start_indice ", start_epoca)        
    
    print("Num epoche", epochs) 
    
    
    for e in range(start_epoca,epochs):
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
                    
                    f = torch.cat((phi_i,phi_j),1)
                    
                    output = model.fc2(f)
                    
                    l_ij = l_ij.type(torch.LongTensor).to(device)
                    #calcoliamo la loss
                    l = criterion(output, l_ij)

                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    #print("numero elementi nel batch ",n)
                    global_step += n
                    
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    acc = accuracy_score(l_ij.to('cpu'),output.to('cpu').max(1)[1])
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
        
        
        figure = plt.figure(figsize=(12,8))
        plt.plot(array_glb_train,array_loss_train)
        plt.plot(array_glb_valid,array_loss_valid)
        plt.xlabel('samples')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotLoss_'+version+'.png')
        plt.clf()
        plt.close(figure)
        
         
        
        
        #conserviamo i pesi del modello alla fine di un ciclo di training e test
        net_save(epochs, model, optimizer, last_loss_train ,last_loss_val,last_acc_train,last_acc_val, global_step_train,global_step_val,'%s.pth'%(exp_name+"_dict"))
        torch.save(model,'%s.pth'%exp_name)
        torch.save(model,directory+"//"+version+"//"+'%s.pth'%(exp_name+"_"+str(e)))
        
        saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,len(train_loader),array_accuracy_train[-1],array_accuracy_valid[-1], array_loss_train[-1],array_loss_valid[-1])
        saveArray(directory,version,array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train,array_glb_valid)
        
        
    f='{:.7f}'.format(tempo.stop())
    return model ,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val


def saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,pair,last_acc_train,last_acc_val, last_loss_train,last_loss_val):
    end = timer()
    time = end-start
    hyperparametr = {"indexEpoch":e,"lr":lr, "momentum" : momentum, "numSampleTrain": pair }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": time}
    writeJsonModelEpoca(directory,version,hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
 


