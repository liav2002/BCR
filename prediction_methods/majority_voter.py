__all__ = [
    'MajorityVoter'
]


class MajorityVoter:
    """
    Class to make a majority vote over multiple (5 or more? odd number anyway) classifications
    """

    def __init__(self, prediction_list, threshold=0.5):
        self.predictions = prediction_list
        self.threshold = threshold

    def vote(self):
        """
        Overall prediction

        :return: 1 if more than half predictions are 1s
        """

        if sum(self.predictions) > self.threshold * len(self.predictions):
            return 1
        else:
            return 0
