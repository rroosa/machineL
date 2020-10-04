# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 19:31:15 2020

@author: rosaz
"""

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from ContrastiveLoss import ContrastiveLoss
from AverageValueMeter import AverageValueMeter
from utils.calculate_time import Timer
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torchvision import transforms
from os.path import join
from prova import net_save, controlFile, controlFileCSV
from torch.utils.tensorboard import SummaryWriter
from EmbeddingNet import EmbeddingNet
from DataSetPairCreate import DataSetPairCreate
from DataSetCreate import DataSetCreate
from torch.utils.data import DataLoader
from torch.nn import functional as F
             
def train_continue(directory,version,path,exp_name,name, model,lr, epochs, momentum,batch_size,resize, margin, logdir):
    #definiamo la contrastive loss
    
    print("Continue model")
    directory = directory
    resize = resize
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    siamese_reload = model
    siamese_reload.to(device)
    checkpoint = torch.load(path)
    
    siamese_reload.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    lossTrain = checkpoint['lossTrain']
    lossValid = checkpoint['lossValid']
    
    print('lossTrain',lossTrain)
    print('lossValid',lossValid)
    global_step_train = checkpoint['global_step_train']
    global_step_val = checkpoint['global_step_valid']
    
    accTrain = checkpoint['accTrain']
    accValid = checkpoint['accValid']
    print('accTrain', accTrain)
    print('accValid',accValid)
    
    print("Epoca %s , lossTrain %s , lossValid ,accTarin, accValid, global_step_train %s , global_step_val %s",epoch, lossTrain, lossValid,accTrain, accValid, global_step_train, global_step_val)
    
    print(siamese_reload.load_state_dict(checkpoint['model_state_dict']))
    #model(torch.zeros(16,3,28,28)).shape
    
    #E' possibile accedere a un dizionario contenente tutti i parametri del modello utilizzando il metodo state_dict .
    state_dict = siamese_reload.state_dict()
    print(state_dict.keys())
        
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in siamese_reload.state_dict():
        print(param_tensor, "\t", siamese_reload.state_dict()[param_tensor].size())
            
    
    controlFileCSV()
    #controlFileCSVPair()
    dataSetPair = DataSetPairCreate(resize)
    dataSetPair.controlNormalize()
   
    
    pair_train = dataSetPair.pair_money_train
    #pair_test = dataSetPair.pair_money_test
    pair_validation = dataSetPair.pair_money_val
        
    pair_money_train_loader = DataLoader(pair_train, batch_size=batch_size, num_workers=0, shuffle=True)
    #pair_money_test_loader = DataLoader(pair_test, batch_size=1024, num_workers=0)
    pair_money_val_loader = DataLoader(pair_validation , batch_size = batch_size, num_workers=0)
        
     
    
    
    criterion = ContrastiveLoss(margin)
    optimizer = SGD(siamese_reload.parameters(), lr, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
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
    
    criterion.to(device)# anche la loss va portata sul device in quanto contiene un parametro(m)
    #definiamo un dizionario contenente i loader di training e test
    loader = {
        'train' : pair_money_train_loader,
        'valid' : pair_money_val_loader
    }
    #global_step_train = global_step_train
    #gloabal_step_val = global_step_val
    
    #lossTrain = lossTrain
    #lossValid = lossValid
    timer = Timer()   
    global_step = global_step_val
    
    for e in range(epochs):
        print("Epoca ",e)
        #iteriamo tra due modalità: train e test
        for mode in ['train','valid'] :
            """
            if mode =='train':
                loss_meter.inizializza(lossTrain, global_step_train)
                acc_meter.inizializza(accTrain, global_step_train)
                global_step=global_step_train
            else:
                loss_meter.inizializza(lossValid, global_step_val)
                acc_meter.inizializza(accValid, global_step_val)
                global_step = global_step_val
              """  
                
            siamese_reload.train() if mode == 'train' else siamese_reload.eval()
            with torch.set_grad_enabled(mode=='train'): #abilitiamo i gradienti solo in training
                
                for i, batch in enumerate(loader[mode]):
                    I_i, I_j, l_ij, _, _ = [b.to(device) for b in batch]
                    #img1, img2, label12, label1, label2
                    #l'implementazione della rete siamese è banale:
                    #eseguiamo la embedding net sui due input
                    phi_i = siamese_reload(I_i)#img 1
                    phi_j = siamese_reload(I_j)#img2

                    #calcoliamo la loss
                    l = criterion(phi_i, phi_j, l_ij)
                                        
                    d = F.pairwise_distance(phi_i.to('cpu'), phi_j.to('cpu'))
                    labs = l_ij.to('cpu')
                    #print(len(labs))
                    tensor = torch.clamp( margin-d, min = 0) # sceglie il massimo  # sceglie il massimo -- se è zero allora sono dissimili
                    #print("max",type(tensor))
                    #print("size max tensor ",tensor.size())
                    #print("tentor 1", tensor)
        
                    for el in tensor:
                        if el <= 2: # SIMILI
                            if mode == 'train':
                                prediction_train.append(0)
                            else:
                                prediction_val.append(0)
                        else: # DISSIMILI
                            if mode == 'train':
                                
                                prediction_train.append(1)
                            else:
                                prediction_val.append(1)
                    """
                    if mode=='train':
                        array_loss_train.append(l.item())
                    else:
                        array_loss_valid.append(l.item())
                    """
                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    global_step += n
                    
                    if mode=='train':
                        labels_train.extend(list(labs.numpy()))
                        print("Lunghezza predette TRAIN ", len(prediction_train))
                        print("Lunghezza vere TRAIN ", len(labels_train))
                        acc = accuracy_score(np.array(labels_train),np.array(prediction_train))
                        acc_meter.add(acc,n)
                        
                        
                    else:
                        labels_val.extend(list(labs.numpy()))
                        print("Lunghezza predette VALID ", len(prediction_val))
                        print("Lunghezza vere VALID ", len(labels_val))
                        acc = accuracy_score(np.array(labels_val),np.array(prediction_val))
                        acc_meter.add(acc,n)
                        
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    n = batch[0].shape[0] #numero di elementi nel batch
                    loss_meter.add(l.item(),n)
                    
                                        
                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                    
                    if mode == 'train':
                        lossTrain =  loss_meter.value()
                        global_step_train = global_step
                        array_loss_train.append(lossTrain)
                        array_acc_train.append(acc_meter.value())
                        array_sample_train.append(global_step_train)
                        print("TRAIN- Epoca",e)
                        print("GLOBAL STEP TRAIN",global_step_train )
                        print("LOSS TRAIN", lossTrain)
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
        
        #torch.save(siamese_reload.state_dict(),'%s.pth'%exp_name) # salvare i parametri del modello
        
        net_save(epochs,siamese_reload, optimizer, lossTrain ,lossValid,array_acc_train[-1],array_acc_valid[-1], global_step_train, global_step_val,'%s.pth'%exp_name)
    f='{:.7f}'.format(timer.stop())
    
    
    return siamese_reload, f, array_loss_train, array_loss_valid, array_sample_train, array_sample_valid,  array_acc_train, array_acc_valid,labels_train,prediction_train,labels_val,prediction_val