import tensorflow as tf


class Model(tf.keras.Model):
    def get_loss(self, real, pred):
        return tf.reduce_mean(tf.square(pred - real))

    def train_step(self, inputs, targets):
        # `inputs` and `targets` are dataset elements threaded through a custom `fit` that iterates
        # an `experimental_distribute_dataset`-wrapped `tf.data` dataset. See wala/ML#618.
        assert inputs.shape == (2,)
        assert inputs.dtype == tf.float32
        assert targets.shape == (2,)
        assert targets.dtype == tf.float32
        return self.get_loss(targets, inputs)

    def fit(self, dataset):
        strategy = tf.distribute.MirroredStrategy()
        dist = strategy.experimental_distribute_dataset(dataset)
        for inputs, targets in dist:
            self.train_step(inputs, targets)


ds = tf.data.Dataset.from_tensor_slices(
    (tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[1.0, 1.0], [1.0, 1.0]]))
)
Model().fit(ds)
