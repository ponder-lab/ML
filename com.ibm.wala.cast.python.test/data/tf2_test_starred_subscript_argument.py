# Test a parameter fed by a starred subscript, `compute_loss(pred, *target[i], i)`: the unpack spreads
# one `(label, bboxes)` pair over `label` and `bboxes`, so `label` is the pair's first array, never
# the pair. A loss over a loader's per-scale targets has this shape.
import numpy as np
import tensorflow as tf


def consume_boxes(boxes1, boxes2):
    assert (
        boxes2.shape in [(6, 4, 4, 3, 4), (6, 2, 2, 3, 4)]
        and boxes2.dtype == np.float64
    )
    return boxes1


def compute_loss(pred, label, bboxes, i=0):
    pred_xywh = pred[:, :, :, :, 0:4]
    label_xywh = label[:, :, :, :, 0:4]
    return consume_boxes(pred_xywh, label_xywh)


class Loader:
    def __init__(self):
        self.batch_size = 6
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= 2:
            raise StopIteration
        self.count += 1
        image = np.zeros((self.batch_size, 8, 8, 3))
        label_s = np.zeros((self.batch_size, 4, 4, 3, 7))
        boxes_s = np.zeros((self.batch_size, 5, 4))
        label_m = np.zeros((self.batch_size, 2, 2, 3, 7))
        boxes_m = np.zeros((self.batch_size, 5, 4))
        small = label_s, boxes_s
        medium = label_m, boxes_m
        return image, (small, medium)


def train_step(image_data, target):
    for i in range(2):
        pred = tf.ones((6, 4, 4, 3, 7)) if i == 0 else tf.ones((6, 2, 2, 3, 7))
        compute_loss(pred, *target[i], i)


for image_data, target in Loader():
    train_step(image_data, target)
