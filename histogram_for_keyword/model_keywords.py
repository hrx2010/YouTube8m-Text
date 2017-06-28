#Script regrouping the functions used for training the decoder model using keywords features only

import gensim
import numpy as np 
import json
import enchant
import csv
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf

#PARAMETERS
input_size = 1024
layer1 = 1024
layer2 = 2048
layer3 = 4716

#layer with relu activation
def mlp(input_,input_dim,output_dim,name="mlp"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        b = tf.get_variable('b',[output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        return tf.nn.relu(tf.matmul(input_,w)+b)

#final layer, no activation (yet)
def mlpbis(input_,input_dim,output_dim,name="mlpbis"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
        b = tf.get_variable('b',[output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))    
        return tf.matmul(input_,w)+b
        
def build_model_mlp(X_,_dropout):
    model = mlpnet(X_,_dropout)
    return model

#building the net
def mlpnet(image,_dropout):
    l1 = mlp(image,input_size,layer1,name='l1')
    l1 = tf.nn.dropout(l1,_dropout)
    l2 = mlp(l1,layer1,layer2,name='l2')
    l2 = tf.nn.dropout(l2,_dropout)
    l3 = mlpbis(l2,layer2,layer3,name='l3')
    return l3

def inference(d):
    return tf.nn.sigmoid(d)

#loss_function
def loss(y,d):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=d)

#gives access to the next batch of data
def next_batch(s,e,inputs,labels):
    input_data = inputs[s:e]
    y = labels[s:e]
    return np.array(input_data),np.array(y)

#to evaluate multi-label classification performance, we also use the log_loss
def compute_log_loss(prediction,labels):
    prediction = np.clip(prediction,0.001,0.999)
    log_loss = -1.0 * np.mean((labels * np.log(prediction)) + ((1.0 - labels) * np.log(1.0 - prediction)))
    return log_loss

