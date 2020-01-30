# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 03:20:49 2020

@author: spijk
"""
import os

class Config:
    def __init__(self, mode='conv', feature='mfcc', nfilt=26, nfeat=13, nfft=256, rate=8000):
        self.mode = mode
        self.feature = feature
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '_' + feature +'.p')
        