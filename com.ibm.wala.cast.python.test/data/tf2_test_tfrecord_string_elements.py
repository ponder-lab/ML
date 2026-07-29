# A TFRecordDataset's elements are 0-D string tensors (one serialized record each): the map
# callback's parameter and the iterated element both type as {[] string} (the gpt-2
# parse_example(serialized_example) corpus form).
import os
import tempfile

import tensorflow as tf


def parse_example(serialized_example):
    assert serialized_example.dtype == tf.string
    assert serialized_example.shape.rank == 0
    return tf.strings.length(serialized_example)


def consume(a):
    pass


with tempfile.TemporaryDirectory() as tmp:
    record_path = os.path.join(tmp, "sample.tfrecord")
    with tf.io.TFRecordWriter(record_path) as writer:
        writer.write(b"payload-0")
        writer.write(b"payload-1")

    dataset = tf.data.TFRecordDataset(record_path)
    mapped = dataset.map(parse_example)
    for length in mapped:
        assert length.dtype == tf.int32
    for element in dataset:
        assert element.dtype == tf.string
        assert element.shape.rank == 0
        consume(element)
