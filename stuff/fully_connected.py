import numpy as np

def gradient_descent(x, correct_y, feed_forward, activation_function,
                     derivative_af, param, learning_rate, lamb):

    """gradient_descent(x, correct_u, feed_forward_function, activation_function,
    derivative_activation_function, parameters, learning_rate, regularization_strength)
    """

    m = len(x)
    predicted_y, list_a = feed_forward(x, param, activation_function)
    # No need to : list_a = list_a[:-1]
    # (because it's predicted_y : not include in list_a)

    # Compute error last layer
    error_this_layer = predicted_y - correct_y
    error_params = [error_this_layer]

    # We're doing back prop
    param.reverse()
    list_a.reverse()
    derivative_af.reverse()

    # Not doing error_layer_1
    for count_i, param_i in enumerate(param[:-1]):

        param_i = np.transpose(param_i)
        # why list_a[count_i] and not list_a[count_i+1] ?
        # - i'm glad you ask, because we don't don't have the last a_i (aka result)
        error_this_layer = np.dot(param_i, error_this_layer) * derivative_af[count_i+1](list_a[count_i])

        # We remove bias because no one talk to him :(
        error_this_layer = np.delete(error_this_layer, 0, 0)
        error_params.append(error_this_layer)

    # We're getting them back in the right way
    param.reverse()
    error_params.reverse()
    list_a.reverse()
    derivative_af.reverse()

    length = len(param)
    big_delta = list(range(length))

    for i in range(length):
        a_i_transpose = np.transpose(list_a[i])
        big_delta[i] = np.dot(error_params[i], a_i_transpose)

    # Regularization (not for bias)
    for i in range(length):
        big_delta[i][:, 1:] = big_delta[i][:, 1:] * (1 / m) + lamb * param[i][:, 1:]
        big_delta[i][:, :1] = big_delta[i][:, :1] * (1 / m)

    # Derivative is big_delta so :
    for i in range(length):
        param[i] -= learning_rate * big_delta[i]

    return param


def feed_forward(x, param, activation_function):
    """example to pass ; param of neural network ; activation_function ; nb of example"""

    # m is the nb of example
    # I used len() because it works if it's a list or a np.array,
    # (it will be convert into np in np.transpose if needed to)
    m = len(x)

    # A useful list for back propagation
    list_a_i = []

    a_i = x
    a_i = np.transpose(a_i)

    for i, thetas_i in enumerate(param):
        # adding bias unit
        a_i = np.r_[np.ones((1, m)), a_i]
        list_a_i.append(a_i)

        z_i = np.dot(thetas_i, a_i)
        a_i = activation_function[i](z_i)

    return a_i, list_a_i
