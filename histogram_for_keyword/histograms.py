#Generates keywords features in the form of histograms

import gensim
import numpy as np 
import json
import enchant
import csv
from sklearn.cluster import MiniBatchKMeans
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

#finding to which centroid isthe word closest to
def l1distance(word,centroids):
    t = np.sum(np.absolute(word-centroids),axis=1)
    return np.argmin(t)

#handling command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_size', type=int,default=1024,)
    parser.add_argument('--unique', type=bool,default=False)
    parser.add_argument('--mode', type=str,default='train')
    args = parser.parse_args()
    return args

args = parse_args()

#PARAMETERS
features_size = int(args.features_size) #number of centroids
unique = bool(args.unique) #if True, you are doing clustering on the set of keywords, but with just one instance per keyword
mode = args.mode #We do this clustering on the training data only, but one can choose to use the development dat
if (mode == 'train') or (mode == 'dev'): #we have to handle the vlaidation data differently
    modebis = 'train'
else:
    modebis = 'val'
output_path = 'data/keywords_kmeans_features_{}/{}_part_'.format(mode,mode)#where do you want to save the histograms ?
step = 20 #we split the output histograms into this number of files

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

#loading the centroids previously computed
centroids = np.load('train_kmeans_centroids.npy')
if unique:
    centroids = np.load('train_kmeans_centroids_unique.npy') 
print(centroids.shape)

for a in range(step):
    with open(output_path+'{}.csv'.format(a), 'w') as f:
        for i in range(a*int(len(videos)/step),(a+1)*int(len(videos)/step)):
            video_histo = np.zeros(features_size) #each video is represented by an histogram of size features_size
            if (len(videos[i]) > 1):
                if (i % 10000) == 0:
                    print('Histograms creation done at {} %'.format(i/len(videos)))
                video_keywords = []
                for u in range(1,len(videos[i])):
                    no_spaces = keywords_preprocessing(videos[i][u])
                    for v in range(len(no_spaces)):
                        if (len(no_spaces[v]) > 2) & ((no_spaces[v] != 'the') & (no_spaces[v] != 'and')):
                            if (d.check(no_spaces[v]) == True):
                                video_keywords.append(no_spaces[v])
                for j in range(len(video_keywords)):
                    try:
                        encoded_word = model[video_keywords[j]]
                        label = l1distance(encoded_word,centroids)
                        video_histo[label] += 1
                    except KeyError:
                        continue
            #writing the histogram
            if type(videos[i][0]) == list:
                f.write(videos[i][0][0])
            else:
                f.write(videos[i][0])                
            for j in range(len(video_histo)):
                f.write(','+str(video_histo[j]))
            f.write('\n')