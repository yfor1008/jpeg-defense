#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : read_tfrecord.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2019/12/17 15:39:34
# @Docs   : 从tfrecord中读取数据, 生成对的抗样本保存在tfrecord中, [0,1]
'''

import os
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--tfrecord_path', default='默认值', type=str, help='tfrecord路径')
    parser.add_argument('-s', '--save_path', default='', type=str, help='保存路径')
    parser.add_argument('-j', '--load_jpeg', default=False, type=str2bool, help='是否解码JPEG')
    parser.add_argument('-m', '--mask_file', default='mask.png', type=str, help='图像mask')
    parser.add_argument('-r', '--rescale_size', default=256, type=int, help='图像缩放大小')

    args = parser.parse_args()

    mask = Image.open(args.mask_file)

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
    if args.load_jpeg:
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    else:
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.reshape(image, [image_h, image_w, 3])

    sess = tf.Session()
    with sess.as_default():
        tf.local_variables_initializer().run() # must be init!!!
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(7000):
            name, example, w, h, label = sess.run([image_name, image, image_w, image_h, image_label])

            mask1 = mask.resize((h, w))
            mask1 = np.array(mask1, 'uint8') > 0
            x = np.expand_dims(mask1, axis=2)
            mask1 = np.concatenate([x, x, x], axis=2)

            example = example * mask1

            if args.load_jpeg:
                pass
            else:
                example = np.array(example * 255, dtype=np.uint8)
            img = Image.fromarray(example)
            img = img.resize((args.rescale_size, args.rescale_size), Image.ANTIALIAS)
            img.save(os.path.join(args.save_path, name))
        coord.request_stop()
        coord.join(threads)
