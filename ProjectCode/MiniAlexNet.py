# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:41:04 2020

@author: rosaz
"""


from torch import nn
from train_siamese import train_siamese

class MiniAlexNet(nn.Module):
    
    def __init__(self):
        super(MiniAlexNet, self).__init__()
        #ridefiniamo il modello utilizzando i moduli sequential.
        #ne definiamo due: un "feature extractor", che estrae le feature maps
        #e un "classificatore" che implementa i livelly FC
        self.feature_extractor = nn.Sequential(
            #Conv1
            nn.Conv2d(3, 16, 5, padding=2), #Input: 3 x 28 x 28. Ouput: 16 x 28 x28
            nn.MaxPool2d(2), #Input: 16 x 28 x 28. Output: 16 x 14 x 14
            nn.ReLU(),

            #Conv2
            nn.BatchNorm2d(16), #dobbiamo passare come parametro il numero di mappe di featurein input
            nn.Conv2d(16, 32, 5, padding=2), #Input 16 x 14 x 14. Output: 32 x 14 x 14
            nn.MaxPool2d(2), #Input: 32 x 14 x 14. Output: 32 x 7 x 7
            nn.ReLU(),

            #Conv3
            nn.BatchNorm2d(32), #dobbiamo passare come parametro il numero di mappe di featurein input
            nn.Conv2d(32, 64, 3, padding=1), #Input 32 x 7 x 7. Output: 64 x 7 x 7
            nn.ReLU(),

            #Conv4
            nn.BatchNorm2d(64), #dobbiamo passare come parametro il numero di mappe di featurein input
            nn.Conv2d(64, 128, 3, padding=1), #Input 64 x 7 x 7. Output: 128 x 7 x 7
            nn.ReLU(),

            #Conv5
            nn.BatchNorm2d(128), #dobbiamo passare come parametro il numero di mappe di featurein input
            nn.Conv2d(128, 256, 3, padding=1), #Input 128 x 7 x 7. Output: 256 x 7 x 7
            nn.MaxPool2d(2), #Input: 256 x 7 x 7. Output: 256 x 3 x 3
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(), #i layer di dropout vanno posizionati prima di FC6 e FC7
            #FC6
            nn.BatchNorm1d(4096), #dobbiamo passare come parametro il numero di feature in input
            nn.Linear(4096, 2048), #Input: 256 * 3 * 3
            nn.ReLU(),

            nn.Dropout(),
            #FC7
            nn.BatchNorm1d(2048), #dobbiamo passare come parametro il numero di mappe di feature in input
            nn.Linear(2048, 1024),
            #•nn.ReLU(),

            #FC8
            #nn.BatchNorm1d(1024), #dobbiamo passare come parametro il numero di mappe di feature in input
            #nn.Linear(1024, out_classes)
        )

    def forward(self,x):
        output = self.feature_extractor(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        #Applichiamo le diverse trasformazioni in cascata

        return output