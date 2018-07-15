# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:35:53 2018

@author: Mritunjay
"""

import numpy as np
import tensorflow as tf
import os
import cv2
import glob
from sklearn.utils import shuffle

def load_image(path, image_Size, val_size):
    label = os.listdir(path)
    images = []
    labels =[]
    for clas in label:
        index = label.index(clas)
        path1 = os.path.join(path, clas, '*g')
        files = glob.glob(path1)
        for ind in files:
            image = cv2.imread(ind)
            image = cv2.resize(image, (image_Size, image_Size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label1 = np.zeros(len(label))
            label1[index] = 1.0
            labels.append(label1)
            
    images = np.array(images)
    labels = np.array(labels)
    total_files = len(images)
    train_size = np.int(total_files*(1-val_size))
    Train, Label = shuffle(images, labels)
    train_sample = Train[:train_size]
    train_label = Label[:train_size]
    val_sample = Train[train_size:]
    val_label = Label[train_size:]
    
    return train_sample, train_label, val_sample, val_label
    


 
 
class image_read ():
    def __init__(self, path,image_size,val_size):
        self.path = path
        self.image_size = image_size
        self.val_size = val_size
        self._epochs_done = 0
        self._index_in_epoch = 0
        
    def image_train_test(self):
        self.train, self.train_y, self.test, self.test_y = load_image(self.path,self.image_size,self.val_size)
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.train.shape[0]:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
        assert batch_size <= self.train.shape[0]
        end = self._index_in_epoch
        return self.train[start:end], self.train_y[start:end]
        
        
        
       
        
