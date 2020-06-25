import numpy as np
from PIL import Image
from random import randrange


class Dataset:

    def __init__(self, data, label, **kwargs):

        self.data = data
        self.label = label

        self.memory = [i for i in kwargs if kwargs[i]]

        if kwargs.get("one_dimensional"):
            self.data = self.one_dimensional(self.data)
        if kwargs.get("mean_normalization"):
            self.data = self.mean_normalization(self.data)
        if kwargs.get("feature_scaling"):
            self.data = self.feature_scaling(self.data)

    @staticmethod
    def one_dimensional(data):
        """ Convert your data into one dimensional

        param: data
        return : data modified

        Can be called when you initialize your object
        like : your_object = Dataset( ..., one_dimensional=True)"""

        resolution = data[0].shape
        return np.reshape(data, (len(data), resolution[0] * resolution[1]))

    @staticmethod
    def mean_normalization(data):
        """ Use mean normalization on data

        param: data
        return : data modified

        Can be called when you initialize your object
        like : your_object = Dataset( ..., mean_normalization=True)"""

        # Pick up a random value
        np_max = np_min = np.max(data[0])
        np_mean = 0

        n = len(data)

        for x in data:

            if np.max(x) > np_max:
                np_max = np.max(x)

            if np.min(x) < np_min:
                np_min = np.min(x)

            np_mean += np.mean(x) / n

        data_set_normalize = []
        for x in data:
            x_normalize = (x - np_mean) / (np_max - np_min)
            data_set_normalize.append(x_normalize)

        return np.array(data_set_normalize)

    @staticmethod
    def feature_scaling(data):
        """ Use feature scaling on data

        param: data
        return : data modified

        Can be called when you initialize your object
        like your_object = Dataset( ..., feature_scaling=True) """

        np_max = np.max(data[0])

        for x in data:
            if np.max(x) > np_max:
                np_max = np.max(x)

        data_set_return = []
        for x in data:
            to_add = x / np_max
            data_set_return.append(to_add)

        return data_set_return

    def random_image(self):
        """Return : a random image and its label pick up from the dataset"""
        el_random = randrange(len(self.data))
        # random in spanish is "aleotorio"

        image_aleotorio, label_aleotorio = self[el_random]
        return Image.fromarray(image_aleotorio), label_aleotorio, el_random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[:, item]

    def __repr__(self):
        to_display = ""
        if self.memory:
            to_display = '\n\nThe dataset has been : \n- ' + '\n- '.join(self.memory)
        return f"Dataset of {len(self)} examples labeled " + to_display

    def __add__(self, other):
        new_data = np.concatenate((self.data, other.data))
        new_label = np.concatenate((self.label, other.label), axis=1)

        return Dataset(new_data, new_label, MIXED=True)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
