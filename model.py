# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:40:04 2018

@author: Mritunjay
"""
import numpy as np
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
import pandas as pd

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def cnn_layer(input1,channels,filter_size,filters_no, stride_size, pool_size,pstride_size):
    
    ## We shall define the weights that will be trained using create_weights function.
    weight = weights(shape=[filter_size, filter_size, channels, filters_no])
    ## We create biases using the create_biases function. These are also trained.
    biase = biases(filters_no)
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input1,filter=weight,strides=[1, stride_size, stride_size, 1],padding='SAME')
    layer += biase
    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,ksize=[1, pool_size, pool_size, 1],strides=[1, pstride_size, pstride_size, 1],padding='SAME')
    ##Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    return layer

def flatten(layer):
    layer_shape = layer.get_shape()
    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()
    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer





def final_layer(input1,num_inputs,num_outputs,use_relu=True):
    #Let's define trainable weights and biases.
    weight = weights(shape=[num_inputs, num_outputs])
    biase = biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input1, weight) + biase
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer



def multi_layer_cnn(inp, no_cnn_layer,channel,filter_size,filter_no_list, stride_size,pool_size,pstride_size):
    filter_no = filter_no_list[0]
    lyer = {}
    #lyer = []
    for layer in range(no_cnn_layer):
        globals()["layer_cnn"+str(layer)]= cnn_layer(input1=inp,channels=channel,filter_size=filter_size,filters_no=filter_no,stride_size=stride_size,
               pool_size=pool_size,pstride_size=pstride_size)
        inp = globals()["layer_cnn"+str(layer)]
        channel = filter_no
        if layer < no_cnn_layer-1:
            filter_no = filter_no_list[layer+1]
        lyer["cnn_layer"+str(layer)] = globals()["layer_cnn"+str(layer)]
        #lyer.append(globals()["layer_cnn"+str(layer)])
    return lyer


def multi_layer_dnn(inp, no_dnn_layer,num_outputsl,use_relu=True):
    lyer = {}
    no_input = inp.get_shape()[1:4].num_elements()
    no_output = num_outputsl[0]
    for layer in range(no_dnn_layer):
        globals()["layer_dnn"+str(layer)]= final_layer(input1=inp, num_inputs=no_input,num_outputs= no_output )
        inp = globals()["layer_dnn"+str(layer)]
        no_input = no_output
        if layer < no_dnn_layer-1:
            
            no_output = num_outputsl[layer+1]
        lyer["dnn_layer"+str(layer)] = globals()["layer_dnn"+str(layer)]
        
        #lyer.append(globals()["layer_cnn"+str(layer)])
    return lyer
        
    


def predict(layer,true_y,batch_size,ilr,data,epoch,x,r_seed,model_name,path):
    session = tf.Session()
    seed(r_seed)
    set_random_seed(r_seed+1)
    
    init_op = tf.global_variables_initializer() 
    initop1 = tf.initialize_all_variables()
    y_pred = tf.nn.softmax(layer,name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    y_true_cls = tf.argmax(true_y, dimension=1)
    #step = tf.Variable(0, trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer,labels=true_y)
    cost = tf.reduce_mean(cross_entropy)
    #rate = tf.train.exponential_decay(ilr, step, 1, 0.9999)
    optimizer = tf.train.AdamOptimizer(learning_rate=ilr).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    total_batch = int(len(data.train) / batch_size)
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    session.run(tf.initialize_all_variables())
    for i in range(epoch):
        
        avg_cost = 0
        for obs in range(total_batch):
            
            x_batch, y_true_batch =  data.next_batch(batch_size)
            feed_dict_tr = {x: x_batch, true_y: y_true_batch}
            feed_dict_val = {x: data.test,true_y: data.test_y}
            session.run(optimizer,feed_dict=feed_dict_tr)
            #tr_loss =  session.run(cost, feed_dict=feed_dict_tr)
            #avg_cost += tr_loss/total_batch
            
        val_loss = session.run(cost, feed_dict=feed_dict_val)
        
        tr_acc = session.run(accuracy, feed_dict=feed_dict_tr)
        vl_acc = session.run(accuracy, feed_dict=feed_dict_val)
                
        #msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Training_loss: {2:.3f},  Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        #print(msg.format(obs + 1, tr_acc, avg_cost, vl_acc, val_loss))
        print("epoch " + str(i+1) + " -training_accuracy: " + str(tr_acc)  + " -validation_accuracy: " + str(vl_acc))
        
    #print("starting saving model------")
   # simple_save = tf.saved_model.simple_save
    #simple_save(session,path,inputs={"x":x,"y":y},outputs={"y_true":y_true})
    return session
    
    
    
            
        
    #return tr_cost, tr_acc, val_acc, val_cost



    
    
        
        