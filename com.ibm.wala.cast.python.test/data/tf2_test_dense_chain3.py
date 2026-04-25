import tensorflow as tf


def consume(tensor):
    pass


class LoopedModel(tf.keras.Model):
    def __init__(self):
        super(LoopedModel, self).__init__()
        self.layers_list = [tf.keras.layers.Dense(4) for _ in range(3)]

    def __call__(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


inputs = tf.keras.Input(shape=(4,))
model = LoopedModel()
result = model(inputs)
consume(result)
