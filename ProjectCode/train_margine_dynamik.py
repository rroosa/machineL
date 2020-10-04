# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:01:36 2020

@author: rosaz
"""


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import functional as F
from AverageValueMeter import AverageValueMeter
from utils.calculate_time import Timer
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from torch import nn
from ContrastiveLoss import ContrastiveLoss
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from prova import net_save, writeJsonModelEpoca, addValueJsonModel,saveArray

def train_margine_dynamik(directory,version,model, train_loader, valid_loader,resize, batch_size, exp_name='model_1', lr=0.0001, epochs=10, momentum=0.99, margin=2, logdir='logs', modeLoss=None):
    
    criterion = ContrastiveLoss()
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
    
    # inizializza 
    euclidean_distance_threshold = 1
    
    
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
    
    soglie = []
    
    for e in range(epochs):
        print("Epoca = ",e)
        print("Euclidean_distance_soglia = ",euclidean_distance_threshold)
        # keep track of euclidean_distance and label history each epoch
        training_euclidean_distance_history = []
        training_label_history = []
        validation_euclidean_distance_history = []
        validation_label_history = []
        
        #iteriamo tra due modalità: train e valid
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
                    print("Etichetta reale", l_ij)
                    l_ij = l_ij.type(torch.LongTensor).to(device)
                    
 
                    
                    #calcoliamo la loss
                    l = criterion(phi_i, phi_j,l_ij )

                    #aggiorniamo il global_step
                    #conterrà il numero di campioni visti durante il training
                    n = I_i.shape[0] #numero di elementi nel batch
                    #print("numero elementi nel batch ",n)
                    global_step += n
                    
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    phi_i = model(I_i)#img 1
                    phi_j = model(I_j)#img2
                        #distanza euclidea
                    if mode=='train':
                        euclidean_distance = F.pairwise_distance(phi_i, phi_j)
                        training_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same (progression) 
                        #equals = training_label.int() == l_ij.int() # 1 if true
                        
                        training_label = training_label.int()
                        acc = accuracy_score(l_ij.to('cpu'),torch.Tensor.numpy(training_label.cpu()))
                        
                        # save euclidean distance and label history 
                        euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
                        training_euclidean_distance_history.extend(euclid_tmp)
                        
                        label_tmp = torch.Tensor.numpy(l_ij.to('cpu'))
                        training_label_history.extend(label_tmp)
                        
            
                    
                    elif mode=='valid':
                        
                        # evaluate validation accuracy using a Euclidean distance threshold
                        euclidean_distance = F.pairwise_distance(phi_i, phi_j)
                        validation_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same
                        #equals = validation_label.int() == l_ij.int() # 1 if true
                        
                        
                        validation_label= validation_label.int()
                        acc = accuracy_score(l_ij.to('cpu'),torch.Tensor.numpy(validation_label.cpu()))
                        
                        # save euclidean distance and label history 
                        euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
                        validation_euclidean_distance_history.extend(euclid_tmp)
                       
                        label_tmp = torch.Tensor.numpy(l_ij.cpu())
                        validation_label_history.extend(label_tmp)
                        
                       
                            

                    n = batch[0].shape[0]
                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)
                    #loggiamo i risultati iterazione per iterazione solo durante il training
                    if mode=='train':
                        writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
                    #una volta finita l'epoca (sia nel caso di training che valid, loggiamo le stime finali)
            
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

        # fine di una epoca
        
        
        print("Loss TRAIN",array_loss_train)
        print("Losss VALID",array_loss_valid)
        print("Accuracy TRAIN",array_accuracy_train)
        print("Accuracy VALID",array_accuracy_valid)
        print("dim acc train",len(array_accuracy_train))
        print("dim acc valid",len(array_accuracy_valid))
        plt.figure(figsize=(12,8))
        plt.plot(array_accuracy_train)
        plt.plot(array_accuracy_valid)
        plt.xlabel('samples')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotAccuracy_'+version+'.png')
        plt.show()
        
        plt.figure(figsize=(12,8))
        plt.plot(array_loss_train)
        plt.plot(array_loss_valid)
        plt.xlabel('samples')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['Training','Valid'])
        plt.savefig(directory+'//plotLoss_'+version+'.png')
        plt.show()
        
        
        euclidean_distance_threshold = aggiusta_soglia(training_label_history, training_euclidean_distance_history,validation_label_history, validation_euclidean_distance_history )
        soglie.append(euclidean_distance_threshold)
         
        saveArray(directory,version,array_loss_train, array_loss_valid, array_accuracy_train, array_accuracy_valid, array_glb_train,array_glb_valid,soglie)
        
        
        saveinFileJson(start,directory,version,resize,batch_size,e, lr, momentum,len(train_loader),array_accuracy_train[-1],array_accuracy_valid[-1], array_loss_train[-1],array_loss_valid[-1])
        addValueJsonModel(directory+"//"+"modelTrained.json",version,"euclidean_distance_threshold","last", euclidean_distance_threshold)
        #writer.add_embedding(phi_i, batch[3], I_i, global_step=global_step, tag=exp_name+'_embedding')
        #conserviamo i pesi del modello alla fine di un ciclo di training e test
        net_save(epochs, model, optimizer, last_loss_train ,last_loss_val,last_acc_train,last_acc_val, global_step_train,global_step_val,'%s.pth'%(exp_name+"_dict"),dict_stato_no=True)
        torch.save(model,'%s.pth'%exp_name)
        torch.save(model,directory+"//"+version+"//"+'%s.pth'%(exp_name+"_"+str(e)))
        
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
 



def aggiusta_soglia(training_label_history, training_euclidean_distance_history,validation_label_history, validation_euclidean_distance_history ):
            # training euclidean distance stats
        """    
        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0t = torch.mean(euclid_if_0) 
        std_euclid_0t = torch.std(euclid_if_0) # population stdev
        mean_euclid_1t = torch.mean(euclid_if_1)
        std_euclid_1t = torch.std(euclid_if_1) # population stdev
        euclid_diff_t = mean_euclid_1t - mean_euclid_0t
        """
        # validation euclidean distance stats
        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0v = torch.mean(torch.Tensor(euclid_if_0)).item()
        std_euclid_0v = torch.std(torch.Tensor(euclid_if_0)).item() # population stdev
        print("Media 0=",mean_euclid_0v)
        print("Std 0=",std_euclid_0v)
        mean_euclid_1v = torch.mean(torch.Tensor(euclid_if_1)).item()
        std_euclid_1v = torch.std(torch.Tensor(euclid_if_1)).item() # population stdev
        
        print("Media 1=",mean_euclid_1v)
        print("Std 1=",std_euclid_1v)
        euclid_diff_v = mean_euclid_1v - mean_euclid_0v

        # after the epoch is completed, adjust the euclidean_distance_threshold based on the validation mean euclidean distances
        euclidean_distance_threshold = (mean_euclid_0v + mean_euclid_1v) / 2

        return euclidean_distance_threshold

