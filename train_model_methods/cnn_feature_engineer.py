"""
@author: Jonah Hess. 2022. All rights reserved.
"""

from librosa.feature import melspectrogram

__all__ = ['CNNFeatureEngineer']


class CNNFeatureEngineer:
    """
    Description: This class is used to engineer the feature of the audio file.
    """

    RATE = 44100  # Audio file sample rate. All recordings are 44100 Hz.
    FRAME = 512  # Audio file frame size.

    def __init__(self, label=None):
        """
            Description: The constructor of the class.
            :param: label: The label of the audio file.
        """

        if label is None:
            self.label = ''
        else:
            self.label = label

    def feature_engineer(self, audio_data):
        """
        Description: Extract features using librosa.feature.
                     Each signal is cut into frames, features are computed for each frame and averaged.
                     The numpy array is transformed into a data frame with named columns.
        :param: audio_data: The input signals samples with frequency 44100 Hz.
        :return: a numpy array (numOfFeatures x numOfShortTermWindows).
                 None - if error occurred.
        """

        try:
            mel_spec = melspectrogram(y=audio_data)
            return mel_spec, self.label

        except ValueError as e:
            print(e)
            return None, None
