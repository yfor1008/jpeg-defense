#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : raodong_ana.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2019/12/26 15:17:59
# @Docs   : 扰动分析
'''

import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--images_path', default='', type=str, help='图像目录')
    parser.add_argument('-k', '--mask_file', default='mask.png', type=str, help='图像mask')
    args = parser.parse_args()

    # images
    images = os.listdir(args.images_path)
    # mask
    mask = Image.open(args.mask_file)

    img_orig = None
    img_fgsm = None
    img_df = None

    for img in images:
        if 'orig' in img:
            img_orig = Image.open(os.path.join(args.images_path, img))
        elif 'fgsm' in img:
            img_fgsm = Image.open(os.path.join(args.images_path, img))
        elif 'df' in img:
            img_df = Image.open(os.path.join(args.images_path, img))

    h, w = img_orig.size

    plt.imshow(img_orig)

    # mask = mask.resize((h, w))
    # mask = np.array(mask, 'uint8') > 0
    # # x = np.expand_dims(mask, axis=2)
    # # mask = np.concatenate([x, x, x], axis=2)

    # img_fgsm = img_fgsm.resize((h, w))
    # img_df = img_df.resize((h, w))

    # img_orig = img_orig.convert('L')
    # img_fgsm = img_fgsm.convert('L')
    # img_df = img_df.convert('L')

    # img_orig = np.array(img_orig, dtype=float)
    # img_fgsm = np.array(img_fgsm, dtype=float)
    # img_df = np.array(img_df, dtype=float)

    # diff1 = (img_fgsm - img_orig) * mask
    # diff2 = (img_df - img_orig) * mask

    # fig, ax = plt.subplots()
    # plot = ax.contourf(diff1)
    # cbar = fig.colorbar(plot)

    plt.show()
