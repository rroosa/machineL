# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:20:58 2020

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
from matplotlib import pyplot as plt
from prova import net_save, writeJsonModelEpoca, saveArray
from os.path import join

def train_class(directory,version,model, train_loader, valid_loader,resize,batch_size, exp_name='experiment', lr=0.01, epochs=10, momentum=0.99, logdir='logs',dizionario= None):
    print("Taining classifacation")
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #writer
    writer = SummaryWriter(join(logdir, exp_name))
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : train_loader,
        'valid' : valid_loader
        }
    
    
    array_accuracy_train = []
    array_accuracy_valid = []    
    array_loss_train = []
    array_loss_valid = []
    array_glb_train = []
    array_glb_valid = []
    
    last_loss_train = 0
    last_loss_val = 0
    last_acc_train = 0
    last_acc_val = 0
    #inizializziamo il global step
    global_step = 0
    tempo = Timer()  
    start = timer()
    start_epoca = 0
    
    
    
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
    start = timer()
    
    print("Num epoche", epochs)
    
    for e in range(start_epoca,epochs):
        print("Epoca= ",e)
        #iteriamo tra due modalità: train e test
        for mode in ['train','valid']:
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                
                for i, batch in enumerate(loader[mode]):
                    print(batch['label'])
                    
                    #x, y = [b.to(device) for b in batch]
                    x=batch['image'].to(device) #"portiamoli sul device corretto"
                    y=batch['label'].to(device)
                    output = model(x)

                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = x.shape[0] #numero di elementi nel batch
                    print("numero elementi nel batch ",n)
                    global_step += n
                    
                    l = criterion(output,y)
                    
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    print("Etichette predette",output.to('cpu').max(1)[1])
                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    
                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)
                    #loggiamo i risultati iterazione per iterazione solo durante il training
                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                        print("Accuracy Train=",acc_meter.value())
                    #una volta finita l'epoca (sia nel caso di training che test, loggiamo le stime finali)
                    
            if mode == 'train':   
                global_step_train=global_step
                last_loss_train = loss_meter.value()
                last_acc_train = acc_meter.value()
                print("Accuracy Train=",acc_meter.value())
                array_accuracy_train.append(acc_meter.value())
                array_loss_train.append(loss_meter.value())
                array_glb_train.append(global_step) 
                
            else:
                global_step_val = global_step
                last_loss_val = loss_meter.value()
                last_acc_val = acc_meter.value()
                print("Accuracy Valid=",acc_meter.value())
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
        net_save(epochs, model, optimizer, last_loss_train ,last_loss_val,last_acc_train,last_acc_val, global_step_train,global_step_val,'%s.pth'%(exp_name+"_dict"),dict_stato_no = True)
       
        #conserviamo i pesi del modello alla fine di un ciclo di training e test
        torch.save(model,directory+"//"+version+"//"+'%s.pth'%(exp_name+"_"+str(e)))
        torch.save(model,'%s.pth'%(exp_name))
        
        saveArray(directory,version,array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train,array_glb_valid)
        
        saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,len(train_loader),array_accuracy_train[-1],array_accuracy_valid[-1], array_loss_train[-1],array_loss_valid[-1])
        
        
    f='{:.7f}'.format(tempo.stop())    
    return model,f, last_loss_train, last_loss_val, last_acc_train, last_acc_val

def saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,pair,last_acc_train,last_acc_val, last_loss_train,last_loss_val):
    end = timer()
    time = end-start
    hyperparametr = {"indexEpoch":e,"lr":lr, "momentum" : momentum, "numBatchTrain": pair }
    contrastiveLoss = {"lossTrain": last_loss_train, "lossValid":last_loss_val}
    accuracy = {"accuracyTrain":last_acc_train , "accuracyValid":last_acc_val }
    time = {"training": time}
    writeJsonModelEpoca(directory,version,hyperparametr,resize, batch_size, contrastiveLoss, accuracy ,time)
 