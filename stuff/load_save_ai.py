import dill as pickle
import gzip
import os


def load_ai(filepath):
    """
    Load an object ai (=neural network)

    Parameter
    ---------

    filepath : str
        neural network's filepath

    Return
    ------

    out : object ai
        ai object
    """

    with gzip.open(filepath, "rb") as f:
        out = pickle.load(f)
        return out


def save_ai(filepath, ai):
        """
        Load an object ai (=neural network)

        Parameter
        ---------

        filepath : str
            neural network's filepath
            ex : '/home/YOU/Banana_plan/your_neural_network_name'

        Return
        ------
        """
        name = os.path.basename(filepath)
        ai.name = name

        # gzip or bzip2 are not that useful, on average -5% size
        with gzip.open(filepath, "wb") as f:
            pickle.dump(ai, f)
