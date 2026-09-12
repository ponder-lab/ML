import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# A multi-GPU training script at its subject shape (wala/ML#896): a Keras applications backbone
# (`MobileNetV2`, `include_top=False`) inside a Functional model, wrapped by a `Sequential` with a
# softmax `Dense(NUM_CLASS)` head, fed from `ImageDataGenerator.flow_from_directory` through
# `strategy.run`. The backbone is `weights=None` and the images are generated, so the program runs
# anywhere; nothing about the shapes depends on either.


def consume_predictions(p):
    pass


def consume_labels(l):
    pass


def consume_images(i):
    pass


def consume_backbone(b):
    pass


NUM_CLASS = 10
EMB_SIZE = 2
IMG_SIZE = 112
BATCH_SIZE = 8
IMAGES_PER_CLASS = 2  # 20 images: two full batches of 8 and a partial batch of 4.

# The batch axis every traced step sees, appended at trace time: a full and a partial batch trace
# separately, so the axis is feed-dependent (`None` statically) while the class axis stays fixed.
traced_prediction_shapes = []

data_dir = tempfile.mkdtemp()
for cls in range(NUM_CLASS):
    os.makedirs(os.path.join(data_dir, "c%d" % cls))
    for i in range(IMAGES_PER_CLASS):
        img = np.random.randint(0, 255, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tf.keras.preprocessing.image.save_img(
            os.path.join(data_dir, "c%d" % cls, "%d.png" % i), img
        )

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    backbone = applications.mobilenet_v2.MobileNetV2(
        include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    y = backbone(x)
    consume_backbone(y)
    assert y.shape.as_list() == [None, 4, 4, 1280], y.shape
    y = tf.keras.layers.AveragePooling2D()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(EMB_SIZE, activation=None)(y)
    featureExtractor = tf.keras.models.Model(inputs=x, outputs=y)
    model = tf.keras.Sequential(
        [featureExtractor, tf.keras.layers.Dense(NUM_CLASS, activation="softmax")]
    )

    model.build(input_shape=[1, IMG_SIZE, IMG_SIZE, 3])
    optimizer = tf.keras.optimizers.Adam(0.001)

with strategy.scope():
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )

    def compute_loss(labels, predictions):
        consume_predictions(predictions)
        consume_labels(labels)
        # Each trace sees one concrete batch; the analysis reads the axis as the feed-dependent
        # `None` (wala/ML#830), which the two distinct traced batch sizes below arbitrate.
        traced_prediction_shapes.append(predictions.shape.as_list())
        assert predictions.shape.as_list()[1:] == [NUM_CLASS], predictions.shape
        assert predictions.dtype == tf.float32
        assert labels.shape.as_list()[1:] == [NUM_CLASS], labels.shape
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=BATCH_SIZE
        )

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

with strategy.scope():

    def train_step(inputs):
        images, labels = inputs
        consume_images(images)
        assert images.shape.as_list()[1:] == [IMG_SIZE, IMG_SIZE, 3], images.shape

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss


with strategy.scope():

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    train_dataset = iter(train_generator)
    for _ in range(len(train_generator)):
        batch_loss = distributed_train_step(next(train_dataset))
        train_accuracy.reset_states()

# Two batch extents (8 and 4) across the traces, one class extent: the batch axis is feed-dependent.
assert sorted(set(map(tuple, traced_prediction_shapes))) == [
    (4, NUM_CLASS),
    (BATCH_SIZE, NUM_CLASS),
], traced_prediction_shapes
