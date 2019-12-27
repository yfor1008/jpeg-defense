#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : get_mydata.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2019/12/14 15:10:34
# @Docs   : 生成自己的数据
'''

import math
import os
import random
import tarfile

try:
    import urllib.request as urllib
except ImportError:
    import urllib

from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string('raw_data_dir', None, 'Directory path for raw images.')
flags.DEFINE_integer('jpeg_quality', None, 'JPEG compression to be applied to images')
FLAGS = flags.FLAGS

def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label, synset, height, width):
    """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data
        # and compresses the image.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data,channels=3)
        if FLAGS.jpeg_quality is not None:
            self._decode_jpeg = tf.image.decode_jpeg(tf.image.encode_jpeg(self._decode_jpeg, format='rgb', quality=FLAGS.jpeg_quality), channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    """Process a single image file.
    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _process_image_files_batch(coder, output_file, filenames, synsets, labels):
    """Processes and saves list of images as TFRecords.
    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        output_file: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        synsets: list of strings; each string is a unique WordNet ID
        labels: map of string to integer; id for all synset labels
    """
    writer = tf.python_io.TFRecordWriter(output_file)

    for filename, synset in zip(filenames, synsets):
        image_buffer, height, width = _process_image(filename, coder)
        label = labels[synset]
        example = _convert_to_example(filename, image_buffer, label, synset, height, width)
        writer.write(example.SerializeToString())

    writer.close()

def _process_dataset(filenames, synsets, labels, output_directory, prefix, num_shards):
    """Processes and saves list of images as TFRecords.
    Args:
        filenames: list of strings; each string is a path to an image file
        synsets: list of strings; each string is a unique WordNet ID
        labels: map of string to integer; id for all synset labels
        output_directory: path where output files should be created
        prefix: string; prefix for each file
        num_shards: number of chucks to split the filenames into

    Returns:
        files: list of tf-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / num_shards))
    print(len(filenames), len(synsets), num_shards, chunksize)
    coder = ImageCoder()

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize: (shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize: (shard + 1) * chunksize]
        output_file = os.path.join(output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(coder, output_file, chunk_files, chunk_synsets, labels)
        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files

def convert_to_tf_records(raw_data_dir):
    """Convert the Imagenet dataset into TF-Record dumps."""

    # Glob all the image files
    image_files = tf.gfile.Glob(os.path.join(raw_data_dir, '*', '*.jpg'))

    # Get image file synset labels from the directory name
    image_synsets = [os.path.basename(os.path.dirname(f)) for f in image_files]
    # print(image_synsets)

    # Create unique ids for all synsets
    labels = {v: k for k, v in enumerate(sorted(set(image_synsets), key=int))}

    # Create image data
    tf.logging.info('Processing the image data.')
    image_records = _process_dataset(image_files, image_synsets, labels, os.path.join(raw_data_dir, '../tfrecords'), 'image', 1)

def main(argv):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.INFO)

    # Convert the raw data into tf-records
    convert_to_tf_records(FLAGS.raw_data_dir)

if __name__ == '__main__':
    app.run(main)