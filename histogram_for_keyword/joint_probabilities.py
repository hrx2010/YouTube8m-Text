#Gives a matrix which rows are labels and columns are keywords
#values are probabilities of apperance

import gensim
import numpy as np 
import json
import enchant
import csv
import re
import argparse

#setting the seed
np.random.seed(7)

#Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

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
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    return args

args = parse_args()

#PARAMETERS
mode = args.mode #We do this clustering on the training data only, but one can choose to use the development dat
if (mode == 'train') or (mode == 'dev'): #we have to handle the vlaidation data differently
    modebis = 'train'
else:
    modebis = 'val'
alpha = 0.001


#loading the ids and keywords of the videos that we want to consider, in the form of one list per video - its first element is the ID
if modebis == 'train':
    videos = []
    f = open('input-{}.txt'.format(mode),'r')
    for line in f.readlines():
        l = line.rstrip().split(' ')
        l = l[0].split('\t') + l[1:]
        if (len(l) > 1):
            l[1] = l[1][1:]
            l[len(l)-1] = l[len(l)-1][0:len(l[len(l)-1])-1]
        videos.append(l)
    print('Number of videos in the dataset: {}'.format(len(videos)))
#It's different with validation data
if modebis == 'val':
    with open('combined_metadata_val.json', encoding = 'utf-8') as json_data:
        words = json.load(json_data)
    videos = []
    for i in range(len(words)):
        if 'video_id' in words[i].keys():
            l = [words[i]['video_id']]
            if 'keywords' in words[i].keys():
                keywords_raw = words[i]['keywords']
                #print(keywords_raw)
                keywords_temp = keywords_raw[0].split(',')
                for j in range(len(keywords_temp)):
                    l.append(keywords_temp[j])
            videos.append(l)
    print('Number of videos in the dataset: {}'.format(len(videos)))

#loading the labels names
label_names = []
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_names.append(row[1])
print('Number of different labels: {}'.format(len(label_names)))

label_ranks = {} #enter the name, gives you the rank
with open('Labels distribution.csv','rt') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if row[0] != 'Label':
            label_ranks[row[0].strip (" ")] = i
            i += 1
print('Should be equal to the previous number: {}'.format(len(label_ranks)))

#loading the labels numbers
label_numbers = {} #enter name, gives you the number
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_numbers[row[1]] = int(row[0])
print('Should be equal to the 2 previous numbers : {}'.format(len(label_numbers)))

#meta-dictionnary of labels occurences given keywords
label_joint_probs = []
for i in range(len(label_names)):
    label_joint_probs.append({})

#loading the groundtruth labels
real_labels = {} #dictionnary whose keys are videos ids and values are list of list of labels (some labels containing several words...)
if modebis == 'train':
    with open('labels_groundtruth_{}.csv'.format(modebis),'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = []
            for i in range(1,len(row)):
                labels.append(label_names[int(row[i])])
            real_labels[row[0][2:len(row[0])-1]] = labels
else:
    with open('labels_groundtruth_{}.txt'.format(modebis),'rt') as f:
        for line in f:
            temp = line.split(" ")
            name = temp[0]
            #print(name)
            labels = []
            for i in range(1,len(temp)):
                labels.append(label_names[int(temp[i])])
            real_labels[name] = labels
print(len(real_labels))

#initializing 2 dictionnaries of labels and keywords probabilities
keywords_prob ={}
labels_prob = {}
label_names_list = sorted(label_ranks, key=label_ranks.get) #we sort labels by frequency
print(label_names_list[0:10])
for i in range(len(label_names_list)):
    labels_prob[label_names_list[i]] = 0

num_of_labels = 0
num_of_keywords = 0
keywords_clean = []
for i in range(100000):
    if (len(videos[i]) > 1):
        if (i % 10000) == 0:
            print('Keywords processing done at {} %'.format(i/len(videos)))
        video_keywords = []
        for u in range(1,len(videos[i])):
            no_spaces = keywords_preprocessing(videos[i][u])
            for v in range(len(no_spaces)):
                try:
                    try:
                        if (len(no_spaces[v]) > 2) & ((no_spaces[v] != 'the') & (no_spaces[v] != 'and')):            
                            if (d.check(no_spaces[v]) == True) & (len(model[no_spaces[v]]) == 300):
                                if not(no_spaces[v] in keywords_clean):
                                    keywords_clean.append(no_spaces[v])
                                video_keywords.append(no_spaces[v])
                    except ValueError:
                        continue                            
                except KeyError:
                    continue
        num_of_keywords += len(video_keywords)
        if (len(video_keywords) > 0):
            #getting the keywords probabilities
            for w in range(len(video_keywords)):
                if video_keywords[w] in keywords_prob.keys():
                    keywords_prob[video_keywords[w]] += 1
                else:
                    keywords_prob[video_keywords[w]] = 1
            try:
                d_labels = real_labels[videos[i][0]]
                #print(d_labels)
                num_of_labels += len(d_labels)
                #getting the labels probabilities
                for w in range(len(d_labels)):
                    labels_prob[d_labels[w]] += 1
                for j in range(len(d_labels)):
                    #print(label_names[d_labels[j]])
                    for k in range(len(video_keywords)):
                        if video_keywords[k] in label_joint_probs[label_ranks[d_labels[j]]].keys():
                            label_joint_probs[label_ranks[d_labels[j]]][video_keywords[k]] += 1
                        else:
                            label_joint_probs[label_ranks[d_labels[j]]][video_keywords[k]] = 1
            except KeyError:
                continue
                #print('cannot')

#transforming keywords and labels dictionnaries into probabilities
print('Total number of different keywords: {}'.format(len(keywords_prob)))
print('This number should be equal to the previous one: {}'.format(len(keywords_clean)))
print('Total number of keywords: {}'.format(num_of_keywords))
for key in keywords_prob.keys():
    keywords_prob[key] /= num_of_keywords
print('Total number of different labels: {}'.format(len(labels_prob)))
print('Total number of labels: {}'.format(num_of_labels))
for key in labels_prob.keys():
    labels_prob[key] /= num_of_labels
#print(labels_prob)
#print(keywords_prob)

keywords_ordered = sorted(keywords_prob, key=keywords_prob.get)[::-1] #we sort keywords by frequency

#writing the joint probability matrix, with both labels (aka rows) and keywords (aka columns) ordered by frequency
with open('labels_keywords_joint_probabilities_test_clean.csv','w') as f:
    f.write('Label')
    for i in range(len(keywords_ordered)):
        f.write(','+keywords_ordered[i])
    f.write('\n')
    for i in range(100):
        if ((i % 500) == 0):
            print('Matrix written at: {}'.format(i/len(label_names_list)))
        f.write(label_names_list[i])
        all_occ = 0
        for key in label_joint_probs[i].keys():
            all_occ += label_joint_probs[i][key]
        for j in range(len(keywords_ordered)):
            if keywords_ordered[j] in label_joint_probs[i].keys():
                occ = ((label_joint_probs[i][keywords_ordered[j]] + alpha) / (all_occ + alpha*len(labels_prob))) * (labels_prob[label_names_list[i]] / keywords_prob[keywords_ordered[j]])
            else:
                occ = ((alpha) / (all_occ + alpha*len(labels_prob))) * (labels_prob[label_names_list[i]] / keywords_prob[keywords_ordered[j]])                
            occ = occ
            f.write(','+str(occ))
        f.write('\n')
