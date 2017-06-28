# Author: Kingsley Kuan

import tensorflow as tf
import numpy as np

# L2 weight decay for regularization
kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0000001)

def multi_conv(inputs, kernel_sizes, filters, name, is_training=False):
    """
    Creates multiple convolutions of different kernel sizes and filters in
    parallel, and joins them by concatenation.
    """
    with tf.variable_scope(name):
        branches = []

        # Create multiple 1d convolutions (conv1d -> batch norm -> relu)
        for i in range(len(kernel_sizes)):
            branch = tf.layers.conv1d(inputs, filters[i], kernel_sizes[i],
                                      padding='same',
                                      kernel_regularizer=kernel_regularizer,
                                      name='conv{}'.format(i+1))
            branch = tf.layers.batch_normalization(
                branch, training=is_training,
                name='conv{}/batch_normalization'.format(i+1))
            branch = tf.nn.relu(branch, name='conv{}/relu'.format(i+1))
            branches.append(branch)

        # Concatenate multiple convolution branches
        concat = tf.concat(branches, axis=-1, name='concat')
        return concat

def model(inputs, is_training=False):
    # Multiple parallel convolutions
    block1 = multi_conv(inputs, [1, 2, 3, 4, 5, 6, 7, 8],
                        [512, 512, 512, 512, 512, 512, 512, 512],
                        name='block1', is_training=is_training)
    net = tf.reduce_max(block1, axis=1, name='maxpool')

    # Fully connected hidden layer (dense -> batch norm -> relu -> dropout)
    net = tf.layers.dense(net, 4096, kernel_regularizer=kernel_regularizer,
                          name='fc1')
    net = tf.layers.batch_normalization(net, training=is_training,
                                        name='fc1/batch_normalization')
    net = tf.nn.relu(net, name='fc1/relu')
    net = tf.layers.dropout(net, rate=0.5, training=is_training,
                            name='fc1/dropout')

    # Fully connected output layer
    net = tf.layers.dense(net, 4716, kernel_regularizer=kernel_regularizer,
                          name='fc2')
    tf.summary.histogram('summary/fc2', tf.nn.sigmoid(net))
    return net

def inference(inputs):
    # Add sigmoid activation at end of network during inference
    net = model(inputs, False)
    net = tf.nn.sigmoid(net)
    return net

def loss(inputs, labels):
    # Gather all losses
    net = model(inputs, True)
    net_loss = tf.losses.sigmoid_cross_entropy(labels, net)
    regularization_loss = tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = net_loss + regularization_loss

    tf.summary.scalar('summary/net_loss', net_loss)
    tf.summary.scalar('summary/regularization_loss', regularization_loss)
    tf.summary.scalar('summary/total_loss', total_loss)
    return total_loss
