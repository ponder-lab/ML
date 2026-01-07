import tensorflow as tf

# Input(shape, batch_size, name, dtype, sparse)
# sparse is 5th arg (index 4)
tf.keras.Input((10,), None, None, None, True)
