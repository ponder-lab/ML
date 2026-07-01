# A second module that defines its own `class Model`, colliding by simple name with the
# `tf.keras.Model` that `tf2_657_model_call.py` subclasses (wala/ML#657). Mirrors NLPGNN's
# `Linear_Regression.py`, which also defines `class Model(object)`.
class Model(object):
    def __call__(self, inputs):
        return inputs


model = Model()
