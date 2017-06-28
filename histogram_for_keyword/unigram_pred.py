#Predicts video labels, solely based on keywords probabilities to be labels
#Loads the joint probabilities matrix and returns a GAP on the wanted dataset (which default is: "val')

import numpy as np
import os
import csv
import eval_util
import json
import re
import enchant
import argparse

#setting the seed
np.random.seed(7)

#load a spell checking English dictionnary
d = enchant.Dict("en_US")    

#getting rid of numbers, parenthses, caps, spaces between words, and splitting on commas
def keywords_preprocessing(keyword):
    no_commas = re.sub(r'\d+', '',keyword.strip('(').strip(')')).lower().split(',') 
    no_spaces = []
    for j in range(len(no_commas)):
        temp = no_commas[j].split(" ")
        for x in range(len(temp)):
            no_spaces.append(temp[x].strip(" "))
    return no_spaces

#handling command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,default='val')
    args = parser.parse_args()
    return args

args = parse_args()

#PARAMETERS
mode = args.mode #We do this clustering on the training data only, but one can choose to use the development dat
if (mode == 'train') or (mode == 'dev'): #we have to handle the vlaidation data differently
    modebis = 'train'
else:
    modebis = 'val'
English = True #only consider videos with english keywords for computing GAP


#loading the joint probability matrix    
joint_probability = []
ordered_keywords = {}
ordered_labels = []
with open('labels_keywords_joint_probabilities_test_clean.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'Label':
            for i in range(1,len(row)):
                ordered_keywords[row[i]] = i-1
        if row[0] != 'Label':
            ordered_labels.append(row[0])
            label_preds = []
            for i in range(1,len(row)):
                label_preds.append(float(row[i]))
            joint_probability.append(label_preds)
print('Number of labels: {}'.format(len(joint_probability)))
print('Number of keywords: {}'.format(len(joint_probability[0])))
print('Should be equal to the previous number: {}'.format(len(ordered_keywords)))

#loading the video_ids that we want to consider
if modebis == 'train':
    videos = []
    f = open('/mnt/data_h/vc/wangzhe/source_code/youtube-8m-master-meta/meta-train-resplit/input-{}.txt'.format(mode),'r')
    for line in f.readlines():
        l = line.rstrip().split(' ')
        l = l[0].split('\t') + l[1:]
        if (len(l) > 1):
            l[1] = l[1][1:]
            l[len(l)-1] = l[len(l)-1][0:len(l[len(l)-1])-1]
        videos.append(l)
#It's different with validation data
if modebis == 'val':
    with open('/mnt/data_g/vc/kingsley/yt8m_text/data_val/combined_metadata.json', encoding = 'utf-8') as json_data:
        words = json.load(json_data)
    videos = []
    for i in range(len(words)):
        if (i % 5000) == 0:
            print(i)
        if 'video_id' in words[i].keys():
            l = [words[i]['video_id']]
            if 'keywords' in words[i].keys():
                keywords_raw = words[i]['keywords']
                #print(keywords_raw)
                keywords_temp = keywords_raw[0].split(',')
                for j in range(len(keywords_temp)):
                    l.append(keywords_temp[j])
            videos.append(l)

#loading the labels names
label_names = []
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_names.append(row[1])
print('Number of different labels: {}'.format(len(label_names)))

num_classes = len(label_names)
evl = eval_util.EvaluationMetrics(num_class = num_classes,top_k=20)
gaps = []

#loading the labels numbers
label_numbers = {} #enter name, gives you the number
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_numbers[row[1]] = int(row[0])
print('Should be equal to the previous number: {}'.format(len(label_numbers)))

#loading the groundtruth labels
real_labels = {}
if modebis == 'train':
    with open('labels_groundtruth_{}.csv'.format(modebis),'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = []
            for i in range(1,len(row)):
                labels.append(label_names[int(row[i])])
            #print(row[0][1:])
            #print(labels)
            real_labels[row[0][2:len(row[0])-1]] = labels
else:
    with open('labels_groundtruth_{}.txt'.format(modebis),'rt') as f:
        for line in f:
            #print(line)
            #print(type(line))
            temp = line.split(" ")
            name = temp[0]
            #print(name)
            labels = []
            for i in range(1,len(temp)):
                labels.append(label_names[int(temp[i])])
            real_labels[name] = labels
print(len(real_labels))

#loading the labels rank
label_ranks = {} #enter the name, gives you the rank
with open('Labels distribution.csv','rt') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if row[0] != 'Label':
            label_ranks[row[0].strip (" ")] = i
            i += 1
print(label_ranks)
print(len(label_ranks))

#getting the predictions per video
for i in range(1000): #computation time is quite slow so we do it on a fraction of the videos only
    if (len(videos[i]) > 1):
        if (i % 1000) == 0:
            print('Keywords processing done at {} %'.format(i/len(videos)))
        video_keywords = []
        video_dico = {}
        for j in range(len(label_names)):
            video_dico[label_names[j]] = 0
        for u in range(1,len(videos[i])):
            no_spaces = keywords_preprocessing(videos[i][u])
            for v in range(len(no_spaces)):
                if (len(no_spaces[v]) > 2) & ((no_spaces[v] != 'the') & (no_spaces[v] != 'and')):
                    if (d.check(no_spaces[v]) == True):
                        video_keywords.append(no_spaces[v])
        for label in label_names:
            label_rank = label_ranks[label]
            if (label_rank < len(joint_probability)):
                preds = joint_probability[label_rank]
                for u in range(len(video_keywords)):
                    try:
                        video_dico[label] += np.log(preds[ordered_keywords[video_keywords[u]]])
                        #print(video_dico[label])
                    except KeyError:
                        #print('keyword not found')
                        continue
        sorted_keys = sorted(video_dico, key=video_dico.get)[0:len(joint_probability)]
        truth = np.zeros(len(label_names))
        if (type(videos[i][0]) == list):
            d_labels = real_labels[videos[i][0][0]]
        else:
            d_labels = real_labels[videos[i][0]]
        for v in range(len(d_labels)):
            pos = label_numbers[d_labels[v]]
            truth[pos] = 1
        #print(np.sum(truth))
        predict = np.zeros(len(label_names))
        for v in range(len(sorted_keys)):
            name = sorted_keys[v]
            pos = label_numbers[name]
            predict[pos] = video_dico[sorted_keys[v]]
        #print(np.min(predict))
        true_max = np.max(np.sort(predict)[0:len(joint_probability)])
        for v in range(len(predict)):
            if predict[v] != 0:
                predict[v] = true_max/predict[v]
        #print(np.sort(predict)[::-1][0:20])
        if (i % 50) == 0:
            print(np.max(predict))
        if English:
            if (np.sum(predict) > 0):
                evl.accumulate(np.array([predict]),np.array([truth]),0.01)
        else:
            evl.accumulate(np.array([predict]),np.array([truth]),0.01)            

#test the GAP
print('GAP overall: {}'.format(evl.get()['gap']))
gaps.append(evl.get()['gap'])
evl.clear()