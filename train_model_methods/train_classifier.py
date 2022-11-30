"""
@author: Liav Ariel. 2022. All rights reserved.
"""

__all__ = ['TrainClassifier']


class TrainClassifier:
    """
    Description: class to train a classifier of audio_samples.
    """

    def __init__(self, x, y):
        """
        Description: The constructor of the class.
        :param x: inputs.
        :param y: labels.
        """

        self.x = x
        self.y = y

    def train_classifier(self):
        """
        Description: Train random forest classifier.
        :return: pipeline, best_param, best_estimator, perf.
        """

        # TODO: CODE HERE
        return

