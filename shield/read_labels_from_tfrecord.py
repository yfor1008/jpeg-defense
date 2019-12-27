#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : read_labels_from_tfrecord.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2019/12/25 20:42:56
# @Docs   : 从tfrecord中获取labels
'''

import os
import tensorflow as tf
import numpy as np

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--tfrecord_path', default='默认值', type=str, help='tfrecord路径')
    parser.add_argument('-s', '--save_path', default='', type=str, help='保存路径')

    args = parser.parse_args()

    filename_queue = tf.train.string_input_producer([args.tfrecord_path], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature_set = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(serialized_example, features=feature_set)

    image_name = tf.cast(features['image/filename'], tf.string)
    image_h = tf.cast(features['image/height'], tf.int32)
    image_w = tf.cast(features['image/width'], tf.int32)
    image_label = tf.cast(features['image/class/label'], tf.int32)

    images = []
    labels = []

    sess = tf.Session()
    with sess.as_default():
        tf.local_variables_initializer().run() # must be init!!!
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(7000):
            name, label = sess.run([image_name, image_label])
            images.append(name)
            labels.append(label)

        coord.request_stop()
        coord.join(threads)

    with open(args.save_path, 'w') as fw:
        for name, label in zip(images, labels):
            fw.write('%s, %d\n' % (name, label))
