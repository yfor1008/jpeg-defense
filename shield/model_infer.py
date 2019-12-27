#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : model_infer.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2019/12/24 20:27:59
# @Docs   : 对生成对抗样本进行推理
'''

import os
import sys
from constants import *
sys.path.append(BASE_DIR)
from shield.opts import model_checkpoint_map, model_class_map
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

def preprocessing(img, mask, reshaped_size=(256, 256)):
    """切圆
    ### Args:
        - img: H*W*C, PIL image, rgb
        - mask: H*W*3, array, same with reshaped
        - reshaped_size, 2*1, (height, width), tuple
    ### Returns:
        image.
    """

    mask1 = mask.resize(reshaped_size)
    mask1 = np.array(mask1, 'uint8') > 0
    x = np.expand_dims(mask1, axis=2)
    mask1 = np.concatenate([x, x, x], axis=2)

    img = np.array(img.resize(reshaped_size, Image.ANTIALIAS), dtype='float16') / 255.0

    img = img * mask1
    return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--images_path', default='', type=str, help='图像目录')
    parser.add_argument('-s', '--image_size', default=256, type=int, help='图像大小')
    parser.add_argument('-k', '--mask_file', default='mask.png', type=str, help='图像mask')
    parser.add_argument('-m', '--model_name', default='', type=str, help='模型名称')
    args = parser.parse_args()

    Model = model_class_map[args.model_name]
    model_checkpoint_path = model_checkpoint_map[args.model_name]
    # images
    images = os.listdir(args.images_path)
    # mask
    mask = Image.open(args.mask_file)

    # build graph
    with tf.Graph().as_default():
        input = tf.placeholder(tf.float32, [1, args.image_size, args.image_size, 3])
        model = Model(input)
        prob = model.fprop(input)['probs']
        y_pred = tf.argmax(model.fprop(input)['probs'], 1)

        # Initialize the tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1., allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        with sess.as_default():
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            model.load_weights(model_checkpoint_path, sess=sess)

            for img in images:

                im = [preprocessing(Image.open(os.path.join(args.images_path, img)), mask)]
                im = np.array(im, dtype=np.float)

                pre, pro = sess.run([y_pred, prob], {input: im})

                print('%s, %d, %f' % (img, pre[0], pro[0, pre[0]]))
                # print(pre[0])
                # print(pro[0, pre[0]])
                # break

