import tensorflow as tf

# A plain subject function whose modeled-op return feeds another function's parameter: the
# logistic-regression shape. The producer is `tf.nn.softmax(tf.matmul(x, W) + b)`, so the
# consumer's parameter should carry the matmul-derived shape.

num_features = 784
num_classes = 10

W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")


def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_x = tf.ones((256, num_features))
batch_y = tf.ones((256,), dtype=tf.uint8)

pred = logistic_regression(batch_x)
assert pred.shape == (256, num_classes), pred.shape
assert pred.dtype == tf.float32, pred.dtype

acc = accuracy(pred, batch_y)
assert acc.shape == (), acc.shape
