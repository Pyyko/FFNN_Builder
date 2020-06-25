import numpy as np


class AI:

    def __init__(self, shape, activation_function, lamb, learning_rate,
                 epoch, mini_batch, n_train, n_test, param=None, name=None):

        self.shape = shape
        self.activation_function_name = activation_function

        self.lamb = lamb
        self.learning_rate = learning_rate

        self.param = param
        self.name = name

        self.accuracy = None

        self.epoch = epoch
        self.mini_batch = mini_batch
        self.n_train = n_train
        self.n_test = n_test

        sigmoid = lambda examples: 1 / (1 + np.exp(-examples))
        reLU = lambda examples: examples * (examples > 0)

        sigmoid_d = lambda a_i: a_i * (1 - a_i)
        reLU_d = lambda a_i: 1 * (a_i > 0)

        if activation_function.lower() == "sigmoid":
            self.activation_function = [sigmoid] * (len(shape) - 1)
            self.derivative_af = [sigmoid_d] * (len(shape) - 1)

        elif activation_function.lower() == "relu":
            self.activation_function = [reLU] * (len(shape) - 2) + [sigmoid]
            self.derivative_af = [reLU_d] * (len(shape) - 2) + [sigmoid_d]
