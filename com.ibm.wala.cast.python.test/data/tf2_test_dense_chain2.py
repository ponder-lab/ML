import tensorflow as tf


def consume(tensor):
    pass


class ChainedModel(tf.keras.Model):
    def __init__(self):
        super(ChainedModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(4)
        self.layer2 = tf.keras.layers.Dense(2)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


inputs = tf.keras.Input(shape=(3,))
model = ChainedModel()
result = model(inputs)
consume(result)
