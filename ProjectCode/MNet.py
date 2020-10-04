# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:42:09 2020

@author: rosaz
"""


from torch import nn


class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 64, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(128, 256, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(256, 512, 3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2)
                             
        )
        
        self.fc = nn.Sequential(nn.BatchNorm1d(512),
                                nn.Linear(512 ,256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 128)
        )
        
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0],-1)
        output = self.fc(output)
        return output