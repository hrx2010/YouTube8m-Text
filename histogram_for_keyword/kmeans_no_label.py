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

#loading the correspondance between labels names and numbers
label_names = []
with open('label_names.csv','rt') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'label_id':
            label_names.append(row[1])

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

encoded_keywords = [] #list of all keywords
keywords_clean = [] #full list of pre-processed keywords, but we only keep one occurence per word
for i in range(len(videos)):
    if (len(videos[i]) > 1):
        if (i % 10000) == 0:
            print('Keywords treatment done at {} %'.format(i/len(videos)))
        video_keywords = [] #pre-processed keywords of the current video
        for u in range(1,len(videos[i])):
            no_spaces = keywords_preprocessing(videos[i][u])
            for v in range(len(no_spaces)):
                if (len(no_spaces[v]) > 2) & ((no_spaces[v] != 'the') & (no_spaces[v] != 'and')): #filtering short and stop-words   
                    if (d.check(no_spaces[v]) == True):
                        video_keywords.append(no_spaces[v])
        try:
            #laoding the labels of the video
            #in some videos, the video_id is inside a list
            if type(videos[i][0]) == list:
                d_labels = real_labels[videos[i][0][0]]
            else:
                d_labels = real_labels[videos[i][0]]
            #going through keywords and removing those which are labels
            for v in range(len(d_labels)):
                label_is_in_keywords = 1
                for w in range(len(d_labels[v])):
                    if not(d_labels[v][w] in video_keywords):
                        label_is_in_keywords = 0
                        break
                if (label_is_in_keywords == 1):
                    for w in range(len(d_labels[v])):
                        video_keywords.remove(d_labels[v][w])
            #passing keywords through word2vec
            for j in range(len(video_keywords)):
                encoded_word = model[video_keywords[j]]
                encoded_keywords.append(encoded_word)
                if not(video_keywords[j] in keywords_clean):
                    keywords_clean.append(video_keywords[j])
        except KeyError:
            continue

#getting arrays for k-means
encoded_keywords = np.array(encoded_keywords).astype(np.float64)
print(encoded_keywords.shape)  
for i in range(len(keywords_clean)):
    keywords_clean[i] = model[keywords_clean[i]]
keywords_clean = np.array(keywords_clean).astype(np.float64)
print(keywords_clean.shape) 

#clustering on latent space
if unique:
    keywords_set = keywords_clean
else:
    keywords_set = encoded_keywords
kmeans = MiniBatchKMeans(n_clusters = features_size, batch_size = 10*features_size, random_state = 7).fit(keywords_set)
print('Kmeans clustering is done')

#testing
print(kmeans.predict(encoded_keywords[0:20]))

#saving the clusters centers
clusters = kmeans.cluster_centers_
np.save('{}_kmeans_centroids_no_label.npy'.format(mode),clusters)
print('Saving centers done')