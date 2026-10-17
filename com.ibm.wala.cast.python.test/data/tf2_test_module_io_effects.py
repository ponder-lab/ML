# Test for wala/ML#1080: a call of the `time`, `random` or `cv2` module, or a file opened with
# `tf.io.gfile.GFile`, reads or writes state the program does not pass it, the clock, the random
# generator or the file system, and a mod-ref query sees that read or write in the calling function.
import os
import random
import tempfile
import time

import cv2
import numpy as np
import tensorflow as tf


def reads_clock(x):
    return x * time.time()


def draws_python_random(x):
    return x * random.random()


def scales_randomly(x):
    return x * random.uniform(0.8, 1.2)


def reads_image(path):
    return cv2.imread(path)


def writes_image(path, image):
    cv2.imwrite(path, image)


def count_lines(filename):
    with tf.io.gfile.GFile(filename, mode="rb") as f:
        num_lines = 0
        while True:
            data = f.read(65536)
            if not data:
                break
            num_lines += data.count(b"\n")
    return num_lines


def consume_line(line):
    assert isinstance(line, str)
    return line


def iterates_file(f):
    for line in f:
        consume_line(line)


def doubles(x):
    return x * 2.0


directory = tempfile.mkdtemp()
image_path = os.path.join(directory, "image.png")
text_path = os.path.join(directory, "lines.txt")
with open(text_path, "w") as text:
    text.write("a\nb\n")

x = tf.ones([2])
assert reads_clock(x).shape == (2,)
assert draws_python_random(x).shape == (2,)
assert scales_randomly(x).shape == (2,)
writes_image(image_path, np.zeros((4, 4, 3), dtype=np.uint8))
assert reads_image(image_path).shape == (4, 4, 3)
assert count_lines(text_path) == 2
iterates_file(tf.io.gfile.GFile(text_path))
assert doubles(x).shape == (2,)
