# Test `ContainerElements` where a function the call graph does not reach, nested in another function, resizes an attribute of its first parameter, `obj.items`, under whose name a class stores a list (wala/ML#1268).
# The nested function's parameter may be any object, so the class's list may be the one it appends to, and it reports no length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class C:
    def __init__(self, items):
        self.items = items


def outer():
    def helper(obj):
        obj.items.append(1)

    return helper


lst = [x, y]
C(lst)
sink_list(lst)
