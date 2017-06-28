#Takes in raw keywords data, pre-process it and perfom k-means on it

import gensim
import numpy as np 
import json
from sklearn.cluster import MiniBatchKMeans
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
    parser.add_argument('--features_size', type=int,default=1024,)
    parser.add_argument('--unique', type=bool,default=False)
    parser.add_argument('--mode', type=str,default='train')
    args = parser.parse_args()
    return args

args = parse_args()

#PARAMETERS
features_size = args.features_size #number of centroids
unique = args.unique #if True, you are doing clustering on the set of keywords, but with just one instance per keyword
mode = args.mode #We do this clustering on the training data only, but one can choose to use the development data
if (mode == 'train') or (mode == 'dev'): #we have to handle the vlaidation data differently
    modebis = 'train'
else:
    modebis = 'val'

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
    print('Number of videos in the dataset: {}'.format(len(videos)))

encoded_keywords = [] #full list of pre-processed keywords
keywords_clean = [] #full list of pre-processed keywords, but we only keep one occurence per word
for i in range(len(videos)):
    if (len(videos[i]) > 1): #does this video have keywords ?
        if (i % 10000) == 0:
            print('Kmeans done at {} %'.format(i/len(videos)))
        for u in range(1,len(videos[i])):
            no_spaces = keywords_preprocessing(videos[i][u])
            for v in range(len(no_spaces)):
                try:
                    try:
                        if (len(no_spaces[v]) > 2) & ((no_spaces[v] != 'the') & (no_spaces[v] != 'and')): #filtering short and stop-words            
                            if (d.check(no_spaces[v]) == True) & (len(model[no_spaces[v]]) == 300): #spell-check and check if keyword is recognized by word2vec
                                if not(no_spaces[v] in keywords_clean) : 
                                    keywords_clean.append(no_spaces[v])
                                encoded_keywords.append(model[no_spaces[v]])
                    except ValueError:
                        continue                            
                except KeyError:
                    continue

del videos

#passing unique keywords to word2vec
print('Number of unique keywords: {}'.format(len(keywords_clean)))
for i in range(len(keywords_clean)):
    keywords_clean[i] = model[keywords_clean[i]]
#getting arrays for k-means
keywords_clean = np.array(keywords_clean)
encoded_keywords = np.array(encoded_keywords)

#clustering on latent space
if unique:
    keywords_set = keywords_clean
else:
    keywords_set = encoded_keywords
kmeans = MiniBatchKMeans(n_clusters = features_size, batch_size = 10*features_size, random_state = 7).fit(encoded_keywords)
print('Kmeans clustering is done')

#testing
print(kmeans.predict(encoded_keywords[0:20]))

#saving the clusters centers
clusters = kmeans.cluster_centers_
np.save('{}_kmeans_centroids.npy'.format(mode),clusters)
print('Saving centers done')