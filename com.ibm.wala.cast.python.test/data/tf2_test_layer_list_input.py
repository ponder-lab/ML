# Test https://github.com/wala/ML/issues/993: a list of tensors passed to a Keras layer's call
# reaches the layer's `call` with its elements typed. Each variant adds one link of a
# group-dispatching kernel's chain and ends in its own sink.
import tensorflow as tf


def consume_direct(x):
    assert x.shape == (4, 3) and x.dtype == tf.float32


def consume_nested(x):
    assert x.shape == (4, 3) and x.dtype == tf.float32


def consume_sliced(x):
    assert x.shape == (4, 3) and x.dtype == tf.float32


def consume_listed(x):
    assert x.shape == (4, 3) and x.dtype == tf.float32


def consume_looped(x):
    assert x.shape == (4, 3) and x.dtype == tf.float32


def consume_counted(x):
    assert x.shape == (4, 3) and x.dtype == tf.float32


def consume_built(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def consume_dispatched(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Distance(tf.keras.layers.Layer):
    def __init__(self, sink, **kwargs):
        super().__init__(**kwargs)
        self.sink = sink

    def call(self, inputs):
        z_0 = inputs[0]
        z_1 = inputs[1]
        self.sink(z_0)
        return tf.reduce_sum(z_0 - z_1, axis=-1)


class Kernel(tf.keras.layers.Layer):
    def __init__(self, distance, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def call(self, inputs):
        return self.distance(inputs)


class Gate(tf.keras.layers.Layer):
    def __init__(self, subnets, **kwargs):
        super().__init__(**kwargs)
        self.subnets = subnets

    def call(self, inputs):
        inputs_less_group = inputs[0:-1]
        expert_list = [[] for _ in range(len(self.subnets))]
        for inp in inputs_less_group:
            inp = tf.split(inp, [2, 2], 0)
            for i in range(len(self.subnets)):
                expert_list[i].append(inp[i])
        outputs = []
        for i in range(len(self.subnets)):
            outputs.append(self.subnets[i](expert_list[i]))
        return outputs


class Listed(tf.keras.layers.Layer):
    def __init__(self, subnets, **kwargs):
        super().__init__(**kwargs)
        self.subnets = subnets

    def call(self, inputs):
        return self.subnets[0](inputs[0:-1])


class Looped(tf.keras.layers.Layer):
    def __init__(self, subnets, **kwargs):
        super().__init__(**kwargs)
        self.subnets = subnets

    def call(self, inputs):
        outputs = []
        for i in range(2):
            outputs.append(self.subnets[i](inputs[0:-1]))
        return outputs


class Counted(tf.keras.layers.Layer):
    def __init__(self, subnets, **kwargs):
        super().__init__(**kwargs)
        self.subnets = subnets

    def call(self, inputs):
        outputs = []
        for i in range(len(self.subnets)):
            outputs.append(self.subnets[i](inputs[0:-1]))
        return outputs


class Built(tf.keras.layers.Layer):
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def call(self, inputs):
        expert_list = [[] for _ in range(2)]
        for inp in inputs[0:-1]:
            inp = tf.split(inp, [2, 2], 0)
            for i in range(2):
                expert_list[i].append(inp[i])
        return self.kernel(expert_list[0])


class Sliced(tf.keras.layers.Layer):
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def call(self, inputs):
        return self.kernel(inputs[0:-1])


a = tf.ones((4, 3))
b = tf.ones((4, 3))
g = tf.zeros((4, 1), dtype=tf.int32)

Distance(consume_direct)([a, b])
Kernel(Distance(consume_nested))([a, b])
Sliced(Kernel(Distance(consume_sliced)))([a, b, g])
Listed([Kernel(Distance(consume_listed))])([a, b, g])
Built(Kernel(Distance(consume_built)))([a, b, g])
Looped([Kernel(Distance(consume_looped)), Kernel(Distance(consume_looped))])([a, b, g])
Counted([Kernel(Distance(consume_counted)), Kernel(Distance(consume_counted))])(
    [a, b, g]
)
Gate([Kernel(Distance(consume_dispatched)), Kernel(Distance(consume_dispatched))])(
    [a, b, g]
)
