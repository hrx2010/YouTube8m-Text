#Trains a simple neural network to get classification results based on the keywords histograms only

import gensim
import numpy as np 
import json
import enchant
import csv
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
import time
import math
import random
import argparse

import model_keywords
from gap import AveragePrecisionCalculator
import eval_util

#handling command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,default='val')
    parser.add_argument('--epochs', type=int,default=4)
    parser.add_argument('--bs', type=int,default=64)
    args = parser.parse_args()
    return args

args = parse_args()

#HYPER-PARAMETERS
test = args.mode #do you wanna to test on dev set or val set ?
step_train = 1 #number of training files to use
step_test = 1 #number of testing files to use
training_data_path = '/data/keywords_features_kmeans_train/'
testing_data_path = '/data/keywords_features_kmeans_{}/'.format(test)
gap_file_name = 'GAPs_val_english_no_label' #saving the GAPs values epoch after epoch in this file
num_classes = 4716
batch_size = args.bs
English = True
global_step = tf.Variable(0, trainable = False)
starter_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase = True)
nb_epochs = args.epochs
sigma = 0.0001 #input smoothing variance
evl = eval_util.EvaluationMetrics(num_class = num_classes, top_k = 20) #evaluating the GAP


#loading the labels names
label_names = []
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_names.append(row[1])
print('Number of labels: {}'.format(len(label_names)))

#loading the labels numbers
label_numbers = {} #enter name, gives you the number
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_numbers[row[1]] = int(row[0])
print('This should be equal to the previous number: {}'.format(len(label_numbers)))

#loading the groundtruth labels
real_labels = {}
with open('labels_groundtruth_train.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        labels = []
        for i in range(1,len(row)):
            labels.append(label_names[int(row[i])])
        #print(row[0][1:])
        #print(labels)
        real_labels[row[0][2:len(row[0])-1]] = labels
real_labels_val = {}
with open('labels_groundtruth_val.txt','rt') as f:
    for line in f:
        #print(line)
        #print(type(line))
        temp = line.split(" ")
        name = temp[0]
        #print(name)
        labels = []
        for i in range(1,len(temp)):
            labels.append(label_names[int(temp[i])])
        real_labels_val[name] = labels
print('Total number of videos: {}'.format(len(real_labels)))

Xtrain = []
Ytrain = []

#building the training set
for a in range(step_train):
    print('Training set is built at {} %'.format(a/step_train))
    with open(training_data_path+'train_part_{}.csv'.format(a), 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            video_id = row[0]
            labels = real_labels[video_id]
            gt = np.zeros(num_classes)
            for i in range(len(labels)):
                gt[label_numbers[labels[i]]] = 1
            inp = np.array(row[1:],dtype=float)
            #print(type(inp[0]))
            if English:
                if (np.sum(inp) > 0):
                    Xtrain.append(inp)
                    Ytrain.append(gt)
            else:
                    Xtrain.append(inp)
                    Ytrain.append(gt)                

print('Number of training examples: {}'.format(len(Xtrain)))

input_size = Xtrain[0].shape[0]

Xtest = []
Ytest = []

video_ids_dev = []

#building the test set
for a in range(step_test):
    print('Testing set is built at {} %'.format(a/step_test))
    with open(testing_data_path+'{}_part_{}.csv'.format(test,a), 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            video_id = row[0]
            labels = real_labels_val[video_id]
            gt = np.zeros(num_classes)
            for i in range(len(labels)):
                gt[label_numbers[labels[i]]] = 1
            inp = np.array(row[1:],dtype=float)
            if English:
                if (np.sum(inp) > 0):
                    Xtest.append(inp)
                    Ytest.append(gt)
                    video_ids_dev.append(video_id)
            else:
                    Xtest.append(inp)
                    Ytest.append(gt)     
                    video_ids_dev.append(video_id)

print('Number of testing examples: {}'.format(len(Xtest)))

#adding some random noise to smoothen training
# Xtrain += sigma*np.random.randn(len(Xtrain),input_size)
# Xtest += sigma*np.random.randn(len(Xtrain),input_size)

#Setting training
vectors = tf.placeholder(tf.float32,shape = ([None,input_size]))
dropout_f = tf.placeholder("float")
model = model_keywords.build_model_mlp(vectors,dropout_f)
labels = tf.placeholder(tf.float32,shape = ([None,num_classes]),name = 'gt')
loss = model_keywords.loss(labels,model)
t_vars = tf.trainable_variables()
d_vars  = [var for var in t_vars if 'l' in var.name]
batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

gaps = [] #saving the GAP values

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #tf.global_variables_initializer().run()
    # Training cycle
    for epoch in range(nb_epochs):
        total_batch = int(len(Xtrain)/batch_size) #number of batches
        start_time = time.time()
        # Loop over all batches
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            # Fit training using batch data
            input_vectors,y = model_keywords.next_batch(s,e,Xtrain,Ytrain)
            #Xtrain += sigma*np.random.randn(len(Xtrain),input_size)
            _,loss_value,predict = sess.run([optimizer,loss,model], feed_dict = {vectors:input_vectors, labels:y,dropout_f:0.8})

            if (math.isnan(predict.mean())):
                break
            
            if (i % 2000) == 0:
                #print(predict[0][0:50])
                #print(y[0][0:50])
                #print('Epoch {}, batch {}, loss value: {}'.format(epoch, i, loss_value))
                print('Epoch {}, batch {}, log loss on train set: {}'.format(epoch, i, model_keywords.compute_log_loss(predict,y)))

        #print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        duration = time.time() - start_time
        print('Epoch duration: {}'.format(duration))

        #test the model
        input_vectors,y = np.array(Xtest[0:5000]),np.array(Ytest[0:5000])
        #input_vectors += sigma*np.random.randn(len(input_vectors),input_size)
        print('Converted tests into arrays')
        _,loss_value,predict = sess.run([optimizer,loss,model], feed_dict = {vectors:input_vectors, labels:y, dropout_f:0.8})
        # print('Epoch {}, loss value: {}'.format(epoch, loss_value))
        # print('Epoch {}, log loss on test set: {}'.format(epoch, model_keywords.compute_log_loss(predict,y)))
        
        evl.accumulate(predict,y,loss_value)

        #test the GAP
        print('GAP on first 5k test videos: {}'.format(evl.get()['gap']))
        gaps.append(evl.get()['gap'])
        evl.clear()

        if len(gaps)>= 2:
            if gaps[len(gaps)-1] < 0.98*gaps[len(gaps)-2]:
                print('Best GAP :{}'.format(gaps[len(gaps)-2]))
                break

        # m = AveragePrecisionCalculator()
        # predict_f = predict[0:5000].flatten()
        # predict_random = np.random.rand(5000*4716)
        # y_f = y[0:5000].flatten()
        # print('GAP on test set: {}'.format(m.ap_at_n(predict_f,y_f,n=20*5000)))
        # print('GAP with random crap: {}'.format(m.ap_at_n(predict_random,y_f,n=20*5000)))

    np.save('gap_per_epoch_{}.npy'.format(gap_file_name),np.array(gaps))

    del Xtrain
    del Ytrain

    all_gaps = []

    #build the predictions csv file
    with open('predictions_keywords_model_dev.csv','w') as f:
        f.write('VideoID'+','+'LabelConfidencePair'+'\n')
        for a in range(int(len(Xtest)/10000)):
            input_vectors,y = np.array(Xtest[a*10000:(a+1)*10000]),np.array(Ytest[a*10000:(a+1)*10000])
            print('Converted full tests into arrays')
            _,loss_value,predict = sess.run([optimizer,loss,model], feed_dict = {vectors:input_vectors, labels:y, dropout_f:0.8})   
            evl.accumulate(predict,y,loss_value)
            all_gaps.append(evl.get()['gap'])
            # predicted_labels = np.argsort(predict)[::-1]
            # for i in range(len(video_ids_dev)):
            #     f.write(video_ids_dev[i]+',')
            #     for j in range(20):
            #         f.write(str(predicted_labels[j]) + ' ' + str(predict[predicted_labels[j]]))
            #     f.write('\n')
    print('Final GAP on full test set: {}'.format(np.mean(np.array(all_gaps))))



