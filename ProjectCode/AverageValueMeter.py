# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:07:15 2020

@author: rosaz
"""


class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
    
    def add(self, value, num):
        self.sum += value*num
        self.num += num
        print("Valore",value)
        
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None
        
    def inizializza(self,value,gb):
        self.sum= value*gb
        self.num = gb