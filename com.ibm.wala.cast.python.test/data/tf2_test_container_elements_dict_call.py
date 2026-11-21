# Test `ContainerElements` on a dictionary built by a call of `dict`
# (https://github.com/wala/ML/issues/1307). The sink receives one dict.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_dict_call(c):
    assert len(c) == 2 and c["a"] is x and c["b"] is y


sink_dict_call(dict(a=x, b=y))
