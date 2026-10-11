# Test https://github.com/wala/ML/issues/1050: an optimizer calls its learning-rate schedule with a
# float32 step when it applies gradients, beside any direct call the program makes.
import tensorflow as tf
import tensorflow_addons as tfa


class Decay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, warmup_steps):
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        assert step.shape == ()
        assert step.dtype in (tf.int64, tf.float32)
        step = tf.cast(step + 1, tf.float32)
        return tf.minimum(tf.pow(step, -0.5), step * tf.pow(self.warmup_steps, -1.5))


class Minimized(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __call__(self, step):
        assert step.shape == ()
        assert step.dtype == tf.float32
        return 0.01 * tf.ones_like(step)


class Lazy(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __call__(self, step):
        assert step.shape == ()
        assert step.dtype == tf.float32
        return 0.01 * tf.ones_like(step)


class DirectOnly(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __call__(self, step):
        assert step.shape == ()
        assert step.dtype == tf.int64
        return 0.01


var = tf.Variable(1.0)

# A direct call, as a unit test makes, and an optimizer applying gradients.
schedule = Decay(4000)
schedule(tf.constant(10, dtype=tf.int64))
optimizer = tf.keras.optimizers.Adam(schedule)
with tf.GradientTape() as tape:
    loss = var * var
optimizer.apply_gradients(zip(tape.gradient(loss, [var]), [var]))

# `minimize` updates through the same schedule call.
tf.keras.optimizers.SGD(learning_rate=Minimized()).minimize(lambda: var * var, [var])

# TensorFlow Addons' `LazyAdam` is an `Adam`.
lazy = tfa.optimizers.LazyAdam(Lazy())
with tf.GradientTape() as tape:
    loss = var * var
lazy.apply_gradients(zip(tape.gradient(loss, [var]), [var]))

# A schedule no optimizer is given keeps only the steps the program passes.
DirectOnly()(tf.constant(3, dtype=tf.int64))
