# A transformation whose receiver may be either of two datasets, whose views fixed to one receiver
# each are keyed apart (wala/ML#1088, wala/ML#1089).
import tensorflow as tf


def consume(dataset):
    pass


def pick(flag):
    wide = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
    narrow = tf.data.Dataset.from_tensor_slices(tf.ones([8, 3]))
    shuffled = (wide if flag else narrow).shuffle(2)
    for element in shuffled:
        assert element.shape == ((4,) if flag else (3,))
    consume(shuffled)


for flag in (True, False):
    pick(flag)
