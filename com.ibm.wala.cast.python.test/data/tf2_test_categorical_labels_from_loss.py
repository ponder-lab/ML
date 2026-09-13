import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# The class axis of `flow_from_directory` labels under `class_mode="categorical"` is the number of
# class subdirectories on disk, which no forward chase reads. It is fixed backwards by a
# shape-constrained consumer (wala/ML#920): `tf.keras.losses.CategoricalCrossentropy` requires
# `y_true` and `y_pred` to have the same shape (a mismatch raises), so labels reaching such a call
# whose predictions are `(batch, 10)` are `(batch, 10)`. The other generators here are the
# controls and declines: labels reaching no loss, labels reaching two losses of different widths,
# a predictions width the analysis cannot read, the sparse loss (rank-1 integer labels, no class
# axis), and the function form, which the mechanism does not name.

NUM_CLASS = 10
IMG = 32
BATCH = 4


def consume_labels_constrained(x):
    pass


def consume_labels_keyword(x):
    pass


def consume_labels_unconsumed(x):
    pass


def consume_labels_two_widths(x):
    pass


def consume_labels_unread_width(x):
    pass


def consume_labels_sparse(x):
    pass


def consume_labels_function_form(x):
    pass


def consume_labels_self_dependent_predictions(x):
    pass


def consume_labels_rank3_predictions(x):
    pass


def make_dir(num_classes):
    d = tempfile.mkdtemp()
    for c in range(num_classes):
        os.makedirs(os.path.join(d, "c%d" % c))
        for i in range(2):
            img = np.random.randint(0, 255, size=(IMG, IMG, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(
                os.path.join(d, "c%d" % c, "%d.png" % i), img
            )
    return d


# Each generator is its own `flow_from_directory` call site, as in the program this mirrors: the
# recognizer keys membership on the labels allocation of one call site, and a shared wrapper
# function would merge every site into one allocation (and one class_mode union).
datagen = ImageDataGenerator(rescale=1.0 / 255)


# Functional models with a declared `Input`, as in the program this mirrors: their output shape is
# computed from the declared input, not from the batch fed to them.
def classifier(width):
    inp = tf.keras.layers.Input(shape=(IMG, IMG, 3))
    out = tf.keras.layers.Dense(width, activation="softmax")(
        tf.keras.layers.Flatten()(inp)
    )
    return tf.keras.Model(inputs=inp, outputs=out)


model = classifier(NUM_CLASS)
model5 = classifier(5)
# A `Sequential` without an `Input` layer: its output shape depends on the batch fed to it, which is
# this generator's own images, so resolving the predictions re-enters the generator being computed
# and reads nothing; the class axis then stays unresolved (a sound decline, exercised below).
sequential_no_input = tf.keras.Sequential(
    [tf.keras.layers.Flatten(), tf.keras.layers.Dense(NUM_CLASS, activation="softmax")]
)
loss_object = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
sparse_loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Constrained: the labels reach the categorical loss against (batch, 10) predictions.
images, labels = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
predictions = model(images)
assert labels.shape == (BATCH, NUM_CLASS) and predictions.shape == (BATCH, NUM_CLASS)
per_example = loss_object(labels, predictions)
assert per_example.shape == (BATCH,)
consume_labels_constrained(labels)

# The same, by keyword.
images_k, labels_k = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
loss_object(y_true=labels_k, y_pred=model(images_k))
consume_labels_keyword(labels_k)

# Unconsumed: a categorical generator whose labels reach no loss.
images_u, labels_u = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
consume_labels_unconsumed(labels_u)

# Two widths: the labels reach two categorical losses whose predictions disagree; only one branch
# runs (the other would raise), and the analysis sees both, so it declines. The dead branch is
# deliberate and must stay ill-formed for the same reason as the rank-3 case below.
images_t, labels_t = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
if len(labels_t.shape) > 5:
    loss_object(labels_t, model5(images_t))
else:
    loss_object(labels_t, model(images_t))
consume_labels_two_widths(labels_t)

# Unread width: predictions from a `Dense` whose unit count the analysis cannot read.
units = int(os.environ.get("UNITS", str(NUM_CLASS)))
model_env = tf.keras.Sequential(
    [tf.keras.layers.Flatten(), tf.keras.layers.Dense(units)]
)
images_w, labels_w = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
loss_object(labels_w, model_env(images_w))
consume_labels_unread_width(labels_w)

# Sparse: rank-1 integer labels; the sparse loss constrains no class axis and is not named.
images_s, labels_s = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="sparse",
        )
    )
)
sparse_loss(labels_s, model(images_s))
assert labels_s.shape == (BATCH,)
consume_labels_sparse(labels_s)

# Function form: not the class the mechanism names, so the axis stays unresolved.
images_f, labels_f = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
tf.keras.losses.categorical_crossentropy(labels_f, model(images_f))
consume_labels_function_form(labels_f)

# Self-dependent predictions: a model whose output shape is computed from these very images.
images_d, labels_d = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
loss_object(labels_d, sequential_no_input(images_d))
consume_labels_self_dependent_predictions(labels_d)

# Predictions of rank 3. THIS BRANCH NEVER EXECUTES AND MUST STAY ILL-FORMED: the call would raise if
# it ran (a rank-2 `y_true` against a rank-3 `y_pred` is incompatible), and that is the arm under
# test. The whole warrant of the constraint is that the program is well-formed, so a statically
# visible call the program cannot make is the boundary of that warrant, and the recognizer must
# refuse to read a width from it. Making the ranks agree here would keep the fixture running, keep
# the test passing, and leave the arm untested.
images_r, labels_r = next(
    iter(
        datagen.flow_from_directory(
            make_dir(NUM_CLASS),
            target_size=(IMG, IMG),
            batch_size=BATCH,
            class_mode="categorical",
        )
    )
)
if len(labels_r.shape) > 5:
    loss_object(labels_r, tf.stack([model(images_r), model(images_r)], axis=1))
consume_labels_rank3_predictions(labels_r)
