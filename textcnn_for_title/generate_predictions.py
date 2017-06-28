#!/usr/bin/env python3

# Author: Kingsley Kuan

import argparse
import os
import tensorflow as tf
import numpy as np
import data_loader
import model

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generates predictions and features using trained model')

    parser.add_argument('--tfrecord_file',
                        type=str,
                        default='data_val/data.tfrecord',
                        help='TFRecord file containing testing data')

    parser.add_argument('--train_dir',
                        type=str,
                        default='train',
                        help='Directory where trained models are located')

    parser.add_argument('--predictions_file',
                        type=str,
                        default='data_val/predictions.csv',
                        help='File to write predictions to')

    parser.add_argument('--features_file',
                        type=str,
                        default='data_val/features.csv',
                        help='File to write features to')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size during testing')

    parser.add_argument('--num_k',
                        type=int,
                        default=20,
                        help='Top K results to keep')

    args = parser.parse_args()
    return args

def generate_predictions(tfrecord_file,
                         train_dir,
                         predictions_file,
                         features_file,
                         batch_size,
                         num_k):
    ids, vectors, _ = data_loader.inputs([tfrecord_file], batch_size=batch_size,
                                         num_threads=16, capacity=batch_size*4,
                                         num_epochs=1, is_training=False)

    predictions = model.inference(vectors)
    features = tf.get_default_graph().get_tensor_by_name('fc1/relu:0')

    init_op = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint(train_dir))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        with open(predictions_file, 'w') as f1, open(features_file, 'w') as f2:
            f1.write('VideoId,LabelConfidencePairs\n')

            while True:
                try:
                    ids_out, predictions_out, features_out = sess.run(
                        [ids, predictions, features])
                except tf.errors.OutOfRangeError:
                    break

                for i, _ in enumerate(ids_out):
                    f1.write(ids_out[i].decode())
                    f1.write(',')
                    top_k = np.argsort(predictions_out[i])[::-1][:num_k]
                    for j in top_k:
                        f1.write('{} {:5f} '.format(j, predictions_out[i][j]))
                    f1.write('\n')

                    f2.write(ids_out[i].decode())
                    f2.write(',')
                    for j in range(len(features_out[i]) - 1):
                        f2.write('{:6e},'.format(features_out[i][j]))
                    f2.write('{:6e}'.format(features_out[i][-1]))
                    f2.write('\n')

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    args = parse_args()
    generate_predictions(args.tfrecord_file,
                         args.train_dir,
                         args.predictions_file,
                         args.features_file,
                         args.batch_size,
                         args.num_k)
