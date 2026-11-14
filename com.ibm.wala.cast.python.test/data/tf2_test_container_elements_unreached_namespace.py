# Test `ContainerElements` where a function the call graph does not reach resizes an attribute of an object whose class has a base no summary models (wala/ML#1255).
# `types.SimpleNamespace.__init__` stores the global list under `self.items` with no store the analysis sees, so no list reports a length.
import threading
import types

import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 3, len(c)


class Holder(types.SimpleNamespace):
    def grow(self):
        self.items.append(1)


G = [x, y]
h = Holder(items=G)
t = threading.Thread(target=h.grow)
t.start()
t.join()
sink_list(G)
