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
        description='Trains TextCNN model')

    parser.add_argument('--tfrecord_file',
                        type=str,
                        default='data_train/data.tfrecord',
                        help='TFRecord file containing training data')

    parser.add_argument('--train_dir',
                        type=str,
                        default='train',
                        help='Directory to store trained models')

    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='Batch size during training')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to run training for')

    args = parser.parse_args()
    return args

def train(tfrecord_file, train_dir, batch_size, num_epochs):
    _, vectors, labels = data_loader.inputs(
        [tfrecord_file], batch_size=batch_size,
        num_threads=16, capacity=batch_size*4,
        min_after_dequeue=batch_size*2,
        num_epochs=num_epochs, is_training=True)

    loss = model.loss(vectors, labels)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create training op with dependencies on update ops for batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001). \
            minimize(loss, global_step=global_step)

    # Create training supervisor to manage model logging and saving
    sv = tf.train.Supervisor(logdir=train_dir, global_step=global_step,
                             save_summaries_secs=60, save_model_secs=600)

    with sv.managed_session() as sess:
        while not sv.should_stop():
            _, loss_out, step_out = sess.run([train_op, loss, global_step])

            if step_out % 100 == 0:
                print('Step {}: Loss {}'.format(step_out, loss_out))

if __name__ == '__main__':
    args = parse_args()
    train(args.tfrecord_file, args.train_dir, args.batch_size, args.num_epochs)
