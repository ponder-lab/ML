# Test for https://github.com/wala/ML/issues/1069 and https://github.com/wala/ML/issues/1070.
import tensorflow as tf

v = tf.Variable([1.0, 2.0])


def appended(x):
    m = []
    m.append((v, x))
    return m


def added(x):
    m = []
    m += [(v, x)]
    return m


def extended(x):
    m = []
    m.extend([(v, x)])
    return m


x = tf.constant([1.0, 2.0])
for build in (appended, added, extended):
    built = build(x)
    assert len(built) == 1
    assert built[0][0] is v
