import tensorflow as tf
# Input(shape, batch_size, name, dtype, sparse, tensor)
# tensor is 6th arg (index 5)
tf.keras.Input((10,), None, None, None, False, 1)
