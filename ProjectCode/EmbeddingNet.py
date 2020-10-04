# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 01:07:11 2020

@author: rosaz
"""

from torch import nn

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 64, 5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2)
        )
        
        self.fc = nn.Sequential(nn.BatchNorm1d(64 *5 * 5),
                                nn.Linear(64 * 5 * 5, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 2)
        )
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0],-1)
        output = self.fc(output)
        return output