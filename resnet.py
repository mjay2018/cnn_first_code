# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 13:27:39 2018

@author: Mritunjay
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

os.chdir('C:\\imagemodel')
import Dataset_reader
path = 'C:\\jbm\\All_61326\\All_61326\\train_61326'

train_sample, train_label, val_sample, val_label = Dataset_reader.load_image(path=path, image_Size=200, val_size=.2)
train_label = np_utils.to_categorical(train_label,len(os.listdir(path)))
val_label = np_utils.to_categorical(val_label, len(os.listdir(path)))
image_input = Input(shape=(200, 200, 3)) # feel free to change depending on dataset
epochs = 5
batch_size = 8

resnet = ResNet50(input_tensor=image_input, weights='imagenet', include_top=False)
resnet.summary()

#l_layer =  model.get_layer('avg_pool').output



for layer in resnet.layers:
    layer.trainable = False



# our layers - you can add more if you want

x = Flatten()(resnet.output)

x = Dense(1000, activation='relu')(x)
x = Dropout(.5)(x)
x = Dense(200,activation=  'relu')(x)
x = Dropout(.2)(x)

folders = glob.glob(path + '/*')

prediction = Dense(len(folders), activation='softmax')(x)


# create a model object

model = Model(inputs=resnet.input, outputs=prediction)
# view the structure of the model
model.summary()
# tell the model what cost and optimization method to use

model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
        )

train_gen = data_gen.flow(train_sample,train_label, batch_size=batch_size)
test_gen= data_gen.flow(val_sample,val_label, batch_size=batch_size)

fitmodel = model.fit_generator(
        train_gen,
        validation_data=test_gen, 
        epochs=epochs,
        steps_per_epoch=len(train_sample) // batch_size,
        validation_steps=len(val_sample) // batch_size,
        )

model.save('resnet_model.h5') 

def get_confusion_matrix(train_sample,train_label, N=len(train_sample)):
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in data_gen.flow(train_sample,train_label, shuffle=False, batch_size=batch_size * 2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break
    cm = confusion_matrix(targets, predictions)
    return cm

cm = get_confusion_matrix(train_sample,train_label)
vcm = get_confusion_matrix(val_sample,val_label)


print(cm)

valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))

print(valid_cm)