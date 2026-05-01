import tensorflow as tf


def func(a):
    assert isinstance(a, tf.Tensor)


# Each element from `tf.data.TextLineDataset` iteration is a 0-D string tensor
# at runtime. The static analysis must classify `func`'s parameter as a tensor
# (wala/ML#452 reproducer).
dataset = tf.data.TextLineDataset(["/tmp/text_lines0.txt", "/tmp/text_lines1.txt"])
for element in dataset:
    func(element)
