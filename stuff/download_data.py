import gzip
import dill as pickle
import numpy as np


def download_data(path_obj, n, path=None, n_total=None):
    """ Download data from pickle object in the form of
    [np_array_data, np_array_label]

    param: path_obj: path to the pickle object,
           n :(total number of examples)

    optional param:
        path : path to MNIST_dataset.csv
        n_total : total number of examples you want to convert from it

        if pickle MNIST pickle object doesn't exist
        Convert MNIST_dataset.csv into one and save it

    return np_array_data, np_array_label"""

    try:
        # Trying to download the obj
        with gzip.open(path_obj, "rb") as f:
            dataset = pickle.load(f)

        data, label = dataset

    except FileNotFoundError:
        # If failed : create it
        # /!\ Only work for MNIST dataset under .csv

        with open(path) as d:
            data_label = d.read()
        images = data_label.split('\n')

        # data_label is under the form of : "7,0,0,139,238,...,0,0\n2,0,0,0,19,293,..."
        # One image by line ; First character is the label

        list_label = []

        # I will use implementation with arraylist because np.append is too slow

        resolution = (28, 28)
        data = np.zeros((n_total, resolution[0], resolution[1]))

        for i in range(n_total):
            # Convert string to list
            str_image = images[i]
            list_image = str_image.split(',')

            # Remove label
            label_image = list_image[0]
            del list_image[0]

            # Reshape the list
            length = resolution[0] * resolution[1]
            p = resolution[0]
            str_pixels = [list_image[j * length // p: (j + 1) * length // p] for j in range(p)]

            # ['7', '0', '0', ...] to [7, 0, 0, ...]
            pixels = list(str_pixels)
            for count_line, line in enumerate(str_pixels):
                pixels[count_line] = list(map(int, str_pixels[count_line]))

            np_image = np.array(pixels, dtype=np.uint8)

            list_label.append(label_image)
            data[i] = np_image

        # At that time we have self.data into np_array 28*28
        # And list_label into list (ex : list_label[0] = "7")

        label = np.zeros((10, n), dtype=int)
        for count, digit in enumerate(list_label):
            purified_label = np.zeros(10, dtype=int)
            purified_label[int(digit)] = 1

            label[:, count] = purified_label

        entire_dataset = [data, label]

        # And we finally save it
        with gzip.open(path_obj, "wb") as f:
            pickle.dump(entire_dataset, f)

    data, label = data[:n], label[:, :n]

    return data, label
