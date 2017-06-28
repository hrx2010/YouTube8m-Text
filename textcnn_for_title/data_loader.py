# Author: Kingsley Kuan

import os
import tensorflow as tf

# Creates operation to decode TFRecords examples
def decode(filename_queue):
    # Create TFRecords reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Feature keys in TFRecords example
    features = tf.parse_single_example(serialized_example, features={
        'id': tf.FixedLenFeature([], tf.string),
        'vector': tf.FixedLenFeature([], tf.string),
        'label': tf.VarLenFeature(tf.int64)
    })

    video_id = features['id']

    # Decode vector and pad to fixed size
    vector = tf.decode_raw(features['vector'], tf.float32)
    vector = tf.reshape(vector, [-1, 300])
    vector = tf.pad(vector, [[0, 40 - tf.shape(vector)[0]], [0, 0]])
    vector.set_shape([40, 300])

    # Get label index
    label = tf.sparse_to_indicator(features['label'], 4716)
    label.set_shape([4716])
    label = tf.cast(label, tf.float32)

    return video_id, vector, label

# Creates input pipeline for tensorflow networks
def inputs(filenames, batch_size=4, num_threads=4, capacity=128,
           min_after_dequeue=16, num_epochs=None, is_training=True):
    with tf.name_scope('input'):
        # Add TFRecords file to queue
        filename_queue = tf.train.string_input_producer(
            filenames,
            shuffle=is_training,
            num_epochs=num_epochs)

        # Decode TFRecords example
        video_id, vector, label = decode(filename_queue)

        # Create operation to batch multiple TFRecords examples
        # Shuffle batches if training
        if is_training:
            video_ids, vectors, labels = tf.train.shuffle_batch(
                [video_id, vector, label],
                batch_size=batch_size, num_threads=num_threads,
                capacity=capacity, min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=True)
        else:
            video_ids, vectors, labels = tf.train.batch(
                [video_id, vector, label],
                batch_size=batch_size, num_threads=num_threads,
                capacity=capacity, allow_smaller_final_batch=True)

        return video_ids, vectors, labels
