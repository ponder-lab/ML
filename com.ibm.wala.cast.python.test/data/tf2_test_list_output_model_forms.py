# Calling a functional model whose outputs may be several lists (wala/ML#1157): an element is read
# per list, and a list whose element positions are unknown leaves the element unknown.
import tensorflow as tf


def decode(conv_output):
    s = tf.shape(conv_output)
    return tf.reshape(conv_output, (s[0], s[1], s[1], 3, 7))


def consume_mixed_last(x):
    pass


def consume_concatenated(x):
    pass


def consume_starred(x):
    pass


def consume_single_row(x):
    pass


inp = tf.keras.layers.Input([26, 26, 3])
conv = tf.keras.layers.Conv2D(21, 1, strides=2)(inp)
narrow = tf.keras.layers.Conv2D(5, 1, strides=2)(inp)
x = tf.ones([4, 26, 26, 3])
xs = [1, 2, 3]
pick = len(xs) > 2


# Two models of different output counts built at one site, so the analysis may take either's
# outputs for each, and `[-1]` is the last of each sequence. A tuple of one output stays a tuple.
def build(outputs):
    return tf.keras.Model(inp, outputs)


two_outputs = build([conv, decode(conv)])
one_output = build((narrow,))
last = two_outputs(x)[-1]
assert last.shape == (4, 13, 13, 3, 7)
consume_mixed_last(last)
last = one_output(x)[-1]
assert last.shape == (4, 13, 13, 5)
consume_mixed_last(last)

# A list display or a concatenation, whose element positions the concatenation does not record.
concatenated = tf.keras.Model(
    inp, [conv, decode(conv)] if pick else [decode(conv)] + [conv]
)
first = concatenated(x)[0]
assert first.shape in [(4, 13, 13, 21), (4, 13, 13, 3, 7)]
consume_concatenated(first)

# A display with a starred element, whose positions are unknown.
heads = [conv]
starred = tf.keras.Model(inp, [decode(conv), *heads])
second = starred(x)[1]
assert second.shape == (4, 13, 13, 21)
consume_starred(second)

# A list of one output returns the output itself, so `[0]` is its first row.
single = tf.keras.Model(inp, [narrow])
row = single(x)[0]
assert row.shape == (13, 13, 5)
consume_single_row(row)
