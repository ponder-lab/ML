# Test for wala/ML#1103: a `+` over Python strings is string concatenation, not a tensor operation,
# so a path built from a directory literal reaches the function it is passed to as no tensor.
import os
import random
import tempfile

import numpy as np
import tensorflow as tf


def create_path_generator(images_filepath, labels_filepath):
    image_paths = open(images_filepath).readlines()
    image_label_paths = []
    for label_txt in os.listdir(labels_filepath):
        label_path = labels_filepath + "/" + label_txt
        for image_path in image_paths:
            image_label_paths.append((image_path.rstrip(), label_path))
    while True:
        random.shuffle(image_label_paths)
        for i in range(len(image_label_paths)):
            yield image_label_paths[i]


def process_image_label(image_path, label_path):
    assert isinstance(label_path, str)
    label = open(label_path).readlines()
    return np.zeros((4, 4, 3)), np.array([len(label)])


def data_generator(images_txt, labels_dir, batch_size):
    gen = create_path_generator(images_txt, labels_dir)
    while True:
        images = np.zeros(shape=[batch_size, 4, 4, 3])
        for i in range(batch_size):
            image_path, label_path = next(gen)
            image, label = process_image_label(image_path, label_path)
            images[i] = image
        yield images


os.chdir(tempfile.mkdtemp())
os.mkdir("labels")
open("labels/a.txt", "w").write("1 2\n")
open("images.txt", "w").write("a.jpg\n")
batch = next(data_generator("./images.txt", "./labels", 2))
assert batch.shape == (2, 4, 4, 3)


def suffix_tensor(t):
    assert t.dtype == tf.string and t.shape == ()
    return t + "/x"


def consume_suffixed(s):
    assert s.dtype == tf.string and s.shape == ()
    return s


consume_suffixed(suffix_tensor(tf.constant("a")))
