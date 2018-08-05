# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 13:27:39 2018

@author: Mritunjay
"""
import keras
from keras.layers import Input, Lambda, Dense, Flatten,GlobalAveragePooling2D,Conv2D,MaxPooling2D, AvgPool2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, adam,adamax
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import glob
import os





os.chdir('C:\\imagemodel')
import Dataset_reader
path = 'C:\\jbm\\All_61326\\All_61326\\train_61326'

train_sample, train_label, val_sample, val_label = Dataset_reader.load_image(path=path, image_Size=128, val_size=.5)
train_label = np_utils.to_categorical(train_label,len(os.listdir(path)))
val_label = np_utils.to_categorical(val_label, len(os.listdir(path)))
IMAGE_SIZE = [128, 128] # feel free to change depending on dataset
epochs = 5000
batch_size = 32

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


for layer in vgg.layers:
    layer.trainable = False

#x = BatchNormalization()(vgg.output)
#x = Conv2D(1024 ,(3,3),padding='same')(x)
#x = BatchNormalization()(x)
#x = MaxPooling2D(pool_size=(2 ,2),strides=(2,2))(x)
#x = Conv2D(512,(1,2),padding='same')(x)
#x = BatchNormalization()(x)
#x = MaxPooling2D(pool_size=(1 ,2),strides=(1,2))(x)
#x = BatchNormalization()(x)
x = Flatten()(vgg.output)
x = BatchNormalization()(x)
x = Dropout(.2)(x)
x = Dense(2048,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(.2)(x)
x = Dense(2048, kernel_initializer="random_uniform",activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(.2)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(.2)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(.2)(x)


folders = os.listdir(path)

prediction = Dense(len(folders), activation='softmax')(x)


# create a model object

model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()
# tell the model what cost and optimization method to use
optim = SGD(lr=0.00001,momentum=.9,nesterov=True)

model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

from keras.callbacks import ModelCheckpoint, EarlyStopping

best_weights_filepath = 'c://jbm///best_weights.hdf5'
earlyStopping=EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss',  save_best_only=True, mode='auto')
callbacks_list = [saveBestModel]

data_gen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=40,
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


#model.fit_generator(data_gen.flow(train_sample, train_label, batch_size=batch_size),
#                    steps_per_epoch=len(train_sample) // batch_size, epochs=epochs)

fitmodel = model.fit_generator(
        train_gen,
        validation_data=test_gen, 
        epochs=epochs,
        steps_per_epoch=len(train_sample) // batch_size,
        validation_steps=len(val_sample) // batch_size,
        callbacks=callbacks_list,
        
        )


fitmodel = model.fit(
        x = train_sample,
        y = train_label,
        validation_data=(val_sample,val_label), 
        batch_size = batch_size,
        epochs=epochs,
        #steps_per_epoch=len(train_sample) // batch_size,
        #validation_steps=len(val_sample) // batch_size,
        )


model.save('vgg_model.h5')