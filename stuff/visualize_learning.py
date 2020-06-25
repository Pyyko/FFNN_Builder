import numpy as np


def cost_function(lamb, thetas, m, y, corrects_y):
    """ :param

       lamb : lambda (for regularization)
       thetas : (theta1, theta2, ...)
       m : how many y (np.shape(y)[1])
       y : neural network's result
       corrects_y : corrects result

       :return cost function value
       """

    j = (1 / m) * np.sum(corrects_y * (-np.log(y)) + (1 - corrects_y) * (-np.log(1 - y)))

    # Mask bias for regularization
    reg = 0
    for i in thetas:
        mask = np.ones(i.shape)
        mask[:1] = np.zeros((1, i.shape[1]))
        reg += np.sum((i * mask) ** 2)

    reg *= lamb / (2 * m)
    j += reg

    return j


def accuracy(y, corrects_y, n):
    good_job = 0
    for i in range(n):

        if np.argmax(y[:, i]) == np.argmax(corrects_y[:, i]):
            good_job += 1

    return good_job / n
