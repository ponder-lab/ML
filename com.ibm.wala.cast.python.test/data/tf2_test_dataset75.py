# The strategy-dispatched half of the multi_gpu_train reduction (tf2_test_dataset19.py is the
# direct half): the batch tuple reaches `train_step`'s unpack only through
# `strategy.experimental_run_v2(train_step, args=(dataset_inputs,))`. The direct unpack carries
# the tuple's instance in the parameter's points-to set (tf2_test_dataset74.py); the question
# this fixture isolates is whether the indirect dispatch preserves it. Analysis-only, and NOT
# executable even with an `images/` directory supplied: `experimental_run_v2` was removed from
# TensorFlow around 2.5 in favor of `Strategy.run`, so 2.9.3 raises AttributeError at the first
# dispatch. The legacy spelling is deliberate; it is what the vendored subject code uses and what
# the `experimental_run_v2` summary in tensorflow.xml models. The shape expectations mirror
# tf2_test_dataset19.py's asserts, which run against the same generator configuration.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

strategy = tf.distribute.MirroredStrategy()

datagen = ImageDataGenerator(rescale=1.0 / 255)
generator = datagen.flow_from_directory(
    "images", target_size=(112, 112), batch_size=512, class_mode="categorical"
)


def train_step(inputs):
    images, labels = inputs
    return tf.reduce_mean(images) + tf.reduce_mean(labels)


@tf.function
def distributed_train_step(dataset_inputs):
    return strategy.experimental_run_v2(train_step, args=(dataset_inputs,))


for epoch in range(1, 3):
    batchs_per_epoch = len(generator)
    train_dataset = iter(generator)

    for _ in range(batchs_per_epoch):
        dataset_inputs = next(train_dataset)
        batch_loss = distributed_train_step(dataset_inputs)
