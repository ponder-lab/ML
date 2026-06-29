import tensorflow as tf
from gpt2_model import Gpt2

model = Gpt2(2, 8, 2, 16, 10, 100)
model.mirrored_strategy = tf.distribute.MirroredStrategy()
model.global_batch_size = 2
ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
        tf.constant([[1, 1], [1, 1]], dtype=tf.int32),
    )
).padded_batch(2)
for inputs, targets in ds:
    model.distributed_train_step(inputs, targets)
