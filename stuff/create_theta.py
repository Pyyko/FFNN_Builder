import numpy as np
from math import sqrt


def random_thetas_sigmoid(neurons):
    """ :param:
        nb_neurons of first layer, // of second layer, // of third layer, ...
        (do not include bias unit)

       :returns:
        (thetas second layer, thetas third layer;...)"""

    # I will choose an initialisation where there is no too big or too low value
    # for the sigmoid activation function (otherwise it will be really slow to train)

    thetas = []

    for count in range(len(neurons)):

        # If count = 0 so count is False : no thetas for first layer
        if count:
            # Taken from Glorot and Bengio (2010)
            # idk why this particular formula

            fan_in = neurons[count - 1]
            fan_out = neurons[count]

            r = 4 * sqrt(6 / (fan_in + fan_out))

            # + 1 to receive bias unit
            theta = np.random.uniform(-r, r, (fan_out, fan_in + 1))

            thetas.append(theta)

    return thetas


def random_thetas_reLU(neurons):
    """random_thetas_reLU(neurons_by_layer)"""

    thetas = []

    for count in range(len(neurons)):

        if count:
            # idk why this particular formula too
            # it's Xavier and He Normal initialization

            fan_in = neurons[count - 1]
            fan_out = neurons[count]

            mod = np.sqrt(2/(fan_in+fan_out))
            theta = np.random.randn(fan_out, fan_in + 1) * mod

            thetas.append(theta)

    return thetas
