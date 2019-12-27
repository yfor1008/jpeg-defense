#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : adv_predict.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2019/12/10 17:41:55
# @Docs   : 对训练好的模型进行inference
'''

import os, sys
import numpy as np
import time
import logging
# logging.basicConfig(level=logging.INFO, filename='run_log.log', filemode='w', datefmt='%Y-%m-%d, %a, %H:%M:%S', format='%(message)s')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
from nets import nets_factory

tf.flags.DEFINE_string(
    'model_name', '', 'model name for mobile network.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for mobile network.')

tf.flags.DEFINE_string(
    'image_folder', '', 'Path to images.')

tf.flags.DEFINE_string(
    'batch_size', '64', 'Path to images.')

FLAGS = tf.flags.FLAGS
model_name = FLAGS.model_name

def preprocessing(img, mask, reshaped_size=(256, 256)):
    """切圆
    ### Args:
        - img: H*W*C, PIL image, rgb
        - mask: H*W*1, array, same with reshaped
        - reshaped_size, 2*1, (height, width), tuple
        - rgb_means: C*1, (meanR, meanG, meanB), tuple
    ### Returns:
        image.
    """

    img = np.array(img.resize(reshaped_size, Image.ANTIALIAS), dtype='float16') / 255.0

    img[:, :, 0] = img[:, :, 0] * mask
    img[:, :, 1] = img[:, :, 1] * mask
    img[:, :, 2] = img[:, :, 2] * mask
    return img

def main(_):
    network_fn = nets_factory.get_network_fn(model_name, num_classes=(15 - 0), is_training=False)
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        logits, _ = network_fn(x_input)
        gailv = tf.reduce_max(tf.nn.softmax(logits), 1)
        pred = tf.argmax(logits, 1)

        # images
        images = os.listdir(FLAGS.image_folder)

        # mask
        MASK = Image.open(os.path.join(os.path.dirname(__file__), 'mask.png'))
        MASK = MASK.resize((256, 256))
        MASK = np.array(MASK, 'float16') > 0

        # Run computation
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.checkpoint_path)

            # log
            logging.basicConfig(level=logging.INFO, filename='predict_%s.log' % model_name, filemode='w', datefmt='%Y-%m-%d, %a, %H:%M:%S', format='%(message)s')
            logging.info('image, gailv, predict, label')

            image_len = len(images)
            # image_len = 100
            step = int(FLAGS.batch_size)
            for idx in range(0, image_len, step):
                if (step + idx) > image_len:
                    step = image_len - idx
                ims = [Image.open(os.path.join(FLAGS.image_folder, img)) for img in images[idx:step+idx]]
                ims = [preprocessing(im, MASK) for im in ims]
                val, classed = sess.run([gailv, pred], feed_dict={x_input: ims})
                for name, v, c in zip(images[idx:step+idx], val, classed):
                    # print(name, v, c)
                    logging.info('%s, %f, %d' % (name, v, c))

if __name__ == '__main__':
    tf.app.run()
