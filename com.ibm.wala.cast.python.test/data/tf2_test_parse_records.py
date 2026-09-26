# A text file read into a list of lines, sliced into a dataset, and mapped: the map callback's
# `line` is a scalar string tensor, the element of a dataset built from Python strings.
import os
import tempfile

import tensorflow as tf


def consume_line(line):
    return line


def consume_label(label):
    return label


def parse_records(line):
    consume_line(line)
    image_path, image_label = tf.io.decode_csv(line, ["", 0])
    consume_label(image_label)
    return image_path, image_label


path = os.path.join(tempfile.mkdtemp(), "dataset.csv")
with open(path, "w") as f:
    f.write("jpg/image_0001.jpg,0\njpg/image_0002.jpg,1\n")
with open(path) as f:
    dataset_file = f.read().splitlines()
data = tf.data.Dataset.from_tensor_slices(dataset_file)
data = data.map(parse_records)
for p, l in data:
    assert p.shape == () and p.dtype == tf.string
    assert l.shape == () and l.dtype == tf.int32
    break


def consume_readline_element(line):
    return line


def consume_split_piece(piece):
    return piece


with open(path) as f2:
    lines_list = f2.readlines()
data2 = tf.data.Dataset.from_tensor_slices(lines_list)
for element in data2:
    assert element.shape == () and element.dtype == tf.string
    consume_readline_element(element)
    break

pieces = "jpg/image_0001.jpg,0".split(",")
data3 = tf.data.Dataset.from_tensor_slices(pieces)
for piece in data3:
    assert piece.shape == () and piece.dtype == tf.string
    consume_split_piece(piece)
    break
