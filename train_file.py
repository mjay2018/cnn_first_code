# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:50:56 2018

@author: Mritunjay
"""
import os
import numpy as np
import tensorflow as tf
os.chdir("C:\\imagemodel")
import Dataset_reader
import model
import datasets




path = 'C:\\jbm\\All_61326\\All_61326\\train_61326'
classes = os.listdir(path)
#data = datasets.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
data = datasets.read_train_sets(train_path=path, image_size=128, classes=classes,validation_size=.2)
data1= Dataset_reader.image_read(path,128,.2)
data1.image_train_test()

image_size=128
num_channels =3
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, image_size,image_size,num_channels], name='x')



## labels

y_true = tf.placeholder(tf.float32, shape=[None, 6], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

j = model.multi_layer_cnn(inp=x,no_cnn_layer=10,channel=3,filter_size=3,filter_no_list=[32,64,128,64,128,256,512,1024,512,256],stride_size=2,pool_size=2,pstride_size=2)

for key, value in j.items():
    globals()[key]=value
    
flat_layer = model.flatten(globals()[key])


y = model.multi_layer_dnn(flat_layer,4,[128,256,512,256])

for key, value in y.items():
    globals()[key] = value

#final = globals()[key]



final = model.final_layer(globals()[key],num_inputs=globals()[key].get_shape()[1:4].num_elements(),num_outputs=6, use_relu=False )

ses = model.predict(final,true_y=y_true,batch_size=8,ilr=1e-4,data=data1,epoch=70,x=x,r_seed =1000, model_name = 'image_model1',path='C:\\jbm\\model')



#  Prediction

import os,glob,cv2
import sys,argparse
#sess = tf.Session()
#saver = tf.train.import_meta_graph('C:\\jbm\\model\\image_model.ckpt.meta')
#saver.restore(sess, tf.train.latest_checkpoint('c:\\jbm\\model'))
graph = tf.get_default_graph()

test_path = 'C:\\jbm\\All_61326\\All_61326\\test_61326'
name = os.listdir(test_path)
os.chdir(test_path)
prediction = []
ses2.run(tf.initialize_all_variables())
for nm in name:
    image = cv2.imread(nm)
    # Resizing the image to our desired size and
    # preprocessing will be done exactly as done during training
    image = cv2.resize(image, (128, 128),0,0, cv2.INTER_LINEAR)
    #images = np.array(image, dtype=np.uint8)
    images = image.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, 128,128,3)
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x= graph.get_tensor_by_name("x:0")    
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, len(os.listdir(path)))) 
    feed_dict_testing = {x: x_batch}
    result=ses2.run(y_pred, feed_dict=feed_dict_testing)
    prediction.append(result)




 