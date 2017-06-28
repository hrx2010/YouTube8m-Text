#!/usr/bin/env python3

# Author: Kingsley Kuan

import argparse
import json
import numpy as np
import tensorflow as tf
import gensim

def parse_args():
    parser = argparse.ArgumentParser(
        description='Converts titles to word vector embeddings using Word2Vec')

    parser.add_argument('--word2vec_model_file',
                        type=str,
                        default='GoogleNews-vectors-negative300.bin',
                        help='File containing the trained Word2Vec model')

    parser.add_argument('--video_titles_file',
                        type=str,
                        default='data_train/video_titles.json',
                        help='JSON file containing preprocessed video titles')

    parser.add_argument('--video_labels_file',
                        type=str,
                        default='labels.json',
                        help='JSON file containing video labels')

    parser.add_argument('--tfrecord_file',
                        type=str,
                        default='data_train/data.tfrecord',
                        help='TFRecord file to write data to')

    args = parser.parse_args()
    return args

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def vec_to_tfexample(video_id, vector, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'id': _bytes_feature(video_id),
        'vector': _bytes_feature(vector),
        'label': _int64_list_feature(label)
}))

def title_to_tfrecord(word2vec_model_file,
                      video_titles_file,
                      video_labels_file,
                      tfrecord_file):
    with open(video_titles_file, encoding='utf-8') as file:
        titles = json.load(file)

    with open(video_labels_file) as file:
        labels = json.load(file)

    longest_title = 0
    count = 0

    # Load trained Word2Vec model
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_file, binary=True)

    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for video_id, title in titles.items():
            # Convert each word in the title to a word vector embedding
            title_vec = []
            for word in title:
                if word in word2vec:
                    title_vec.append(word2vec[word].astype(dtype=np.float32))

            # Skip video if nothing could be converted
            if len(title_vec) == 0:
                continue

            if len(title_vec) > longest_title:
                longest_title = len(title_vec)

            # Convert word vector embedding into TFRecord format
            title_vec = np.asarray(title_vec, dtype=np.float32)
            example = vec_to_tfexample(video_id.encode(),
                                       title_vec.tobytes(),
                                       labels[video_id])
            writer.write(example.SerializeToString())

            count += 1
            print("Count: {}".format(count), end='\r', flush=True)
        writer.close()

    print("Longest Title: {}".format(longest_title))
    print("Count: {}".format(count))

if __name__ == '__main__':
    args = parse_args()
    title_to_tfrecord(args.word2vec_model_file,
                      args.video_titles_file,
                      args.video_labels_file,
                      args.tfrecord_file)
