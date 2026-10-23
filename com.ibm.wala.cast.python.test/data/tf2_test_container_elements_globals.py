# Test `ContainerElements` where a list is mutated only inside a callback a library summary never
# calls, through a global, and through `getattr`.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_lambda_global(c):
    assert len(c) == 3


def sink_declared_global(c):
    assert len(c) == 3


def sink_getattr(c):
    assert len(c) == 3


G = [x, y]
sorted([1], key=lambda v: G.append(v) or v)
sink_lambda_global(G)

H = [x, y]


def append_to_h(v):
    global H
    H.append(v)
    return v


sorted([1], key=append_to_h)
sink_declared_global(H)

ga = [x, y]
getattr(ga, "append")(y)
sink_getattr(ga)
