import numpy as np
import tensorflow as tf

# The middle arm of the wala/ML#855 three-arm geometry: the same layer applications fed from
# datasets whose element is a SINGLE tensor, so nothing is destructured and the element carries
# real allocations in its points-to set. Separates "the input came from a dataset" from "the input
# came from a destructured tuple element"; representative per input rank, since every candidate
# shares the one argument-read implementation.


def consume_bn_single(x):
    pass


def consume_gap_single(x):
    pass


rows = 8
imgs = tf.constant(np.ones((rows, 16, 16, 3), dtype=np.float32))
seqs = tf.constant(np.ones((rows, 20, 32), dtype=np.float32))

img_ds = tf.data.Dataset.from_tensor_slices(imgs).batch(4, drop_remainder=True)
seq_ds = tf.data.Dataset.from_tensor_slices(seqs).batch(4, drop_remainder=True)

for x_img in img_ds:
    bn = tf.keras.layers.BatchNormalization()(x_img)
    assert bn.shape == (4, 16, 16, 3), bn.shape
    consume_bn_single(bn)

for x_seq in seq_ds:
    gap = tf.keras.layers.GlobalAveragePooling1D()(x_seq)
    assert gap.shape == (4, 32), gap.shape
    consume_gap_single(gap)
