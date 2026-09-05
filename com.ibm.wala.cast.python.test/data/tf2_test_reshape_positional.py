# Witness for the positional -1 reshape fold (crf.py group): the input has a
# non-constant leading axis and a known trailing extent, so the product rule
# declines (it needs the full total) and the -1 placeholder survives. The target
# shape's other axes correspond positionally to the input's, so the -1 must equal
# the input's extent at its position (46), with no total. Before the fold the
# reshape result was (8, 10, ?); after it is (8, 10, 46).
import tensorflow as tf


def consume(x):
    pass


inp = tf.keras.Input(shape=(10, 46))
r = tf.reshape(inp, [8, 10, -1])
consume(r)
