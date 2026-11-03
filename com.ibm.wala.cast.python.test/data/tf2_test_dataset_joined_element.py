# Test https://github.com/wala/ML/issues/1199: a transformation whose receiver may be either of
# two datasets reads each copied member through the dataset it came from.
import tensorflow as tf


def consume_batched(x):
    pass


def consume_shuffled(x):
    pass


def flat_dataset():
    return tf.data.Dataset.from_tensor_slices((tf.ones([8, 6]), tf.zeros([8])))


def pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    )


def batched(flag):
    for x, y in (flat_dataset() if flag else pair_dataset()).shuffle(4).batch(2):
        assert isinstance(x, (tuple, tf.Tensor))
        if isinstance(x, tf.Tensor):
            assert x.shape == (2, 6) and x.dtype == tf.float32
        consume_batched(x)


def shuffled(flag):
    for x, y in (flat_dataset() if flag else pair_dataset()).shuffle(4):
        assert isinstance(x, (tuple, tf.Tensor))
        if isinstance(x, tf.Tensor):
            assert x.shape == (6,) and x.dtype == tf.float32
        consume_shuffled(x)


for flag in (True, False):
    batched(flag)
    shuffled(flag)
