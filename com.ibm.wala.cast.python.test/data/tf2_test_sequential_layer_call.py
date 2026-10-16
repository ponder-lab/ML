import tensorflow as tf

# wala/ML#832: a `Sequential` model's own call is what calls the layers in its list, so a user
# layer inside one receives the previous layer's output. Each spelling of the list below has its
# own user layer class, called only through its model, so each `call`'s input comes from that
# model path alone.


class UpLiteral(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, 3, padding="same")

    def call(self, x):
        assert x.shape == (2, 50, 16), x.shape
        assert x.dtype == tf.float32
        return self.conv(x)


class UpPlus(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, 3, padding="same")

    def call(self, x):
        assert x.shape == (3, 40, 16), x.shape
        assert x.dtype == tf.float32
        return self.conv(x)


class UpLoop(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, 3, padding="same")

    def call(self, x):
        assert x.shape[0] == 4 and x.shape[1] == 30, x.shape
        assert x.dtype == tf.float32
        return self.conv(x)


class UpAdd(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, 3, padding="same")

    def call(self, x):
        assert x.shape == (5, 20, 16), x.shape
        assert x.dtype == tf.float32
        return self.conv(x)


literal = tf.keras.Sequential(
    [tf.keras.layers.Conv1D(16, 3, padding="same"), UpLiteral(8)]
)
assert literal(tf.random.normal([2, 50, 80])).shape == (2, 50, 8)

layers = []
layers += [tf.keras.layers.Conv1D(16, 3, padding="same"), UpPlus(8)]
plus = tf.keras.models.Sequential(layers)
assert plus(tf.random.normal([3, 40, 80])).shape == (3, 40, 8)

stages = [tf.keras.layers.Conv1D(16, 3, padding="same")]
for filters in [16, 16]:
    stages += [UpLoop(filters)]
loop = tf.keras.Sequential(stages)
assert loop(tf.random.normal([4, 30, 80])).shape == (4, 30, 16)

added = tf.keras.Sequential()
added.add(tf.keras.layers.Conv1D(16, 3, padding="same"))
added.add(UpAdd(8))
assert added(tf.random.normal([5, 20, 80])).shape == (5, 20, 8)
