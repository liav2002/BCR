import re

__all__ = ['BabyCryPredictor']

import numpy as np


class BabyCryPredictor:
    """
    Class to classify a new audio file as a baby cry or not.
    """

    def __init__(self, model):
        self.model = model

    def cnn_classify(self, new_signal):
        """
        Classify a new audio file as a baby cry or not.
        :param new_signal: the new audio signal.
        :return: True if the new audio signal is a baby cry, False otherwise.
        """

        category = self.model.predict(new_signal, verbose=0)

        match = np.argmax(category) == 0

        if match:
            return 1
        else:
            return 0

    def svc_classify(self, new_signal):
        """
        Classify a new audio file as a baby cry or not.
        :param new_signal: the new audio signal.
        :return: True if the new audio signal is a baby cry, False otherwise.
        """

        category = self.model.predict(new_signal)

        match = re.search(r'Crying baby', category[0])

        if match:
            return 1
        else:
            return 0
