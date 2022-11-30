"""
@author: Liav Ariel. 2022. All rights reserved.
"""

__all__ = ['FeatureEngineer']


class FeatureEngineer:
    """
    Description: This class is used to engineer the feature of the audio file.
    """

    RATE = 44100  # Audio file sample rate. All recordings are 44100 Hz.
    FRAME = 512  # Audio file frame size.

    """
    Description: The constructor of the class.
    """

    def __init__(self, label=None):
        if label is None:
            self.label = ''
        else:
            self.label = label

    """
    Description: Extract features using librosa.feature.
                 Each signal is cut into frames, features are computed for each frame and averaged.
                 The numpy array is transformed into a data frame with named columns.

    :param: audio_data: The input signals samples with frequency 44100 Hz.
    :return: a numpy array (numOfFeatures x numOfShortTermWindows).
    """

    def feature_engineer(self, audio_data):
        # TODO: CODE HERE
        return

    """
    Description: This method is used to compute feature using librosa methods.
    :param: audio_data: The input signals samples with frequency 44100 Hz.
    :param: feature_name: The name of the feature to compute.
    :return: a numpy array (numOfFeatures x numOfShortTermWindows).
    """

    def compute_librosa_features(self, audio_data, feature_name):
        # TODO: CODE HERE
        return
