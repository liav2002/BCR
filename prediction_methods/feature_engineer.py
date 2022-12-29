import numpy as np
import math

__all__ = ['FeatureEngineer']

from librosa.feature import melspectrogram


class FeatureEngineer:
    """
    Derive Features
    """

    def __init__(self, duration=5):
        self.duration = duration
        self.rate = 44100
        self.frame = 512
        self.numpy_z_shape_relation = 86.2  # for duration of 1 second, the shape of the numpy array is (128, 86.2).

    def cnn_feature_engineer(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged.
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signals samples with frequency 44100 Hz.
        :return: mel spectrogram. (cnn working with 2d data of shape (numOfFeatures x numOfShortTermWindows))
        """

        try:
            mel_spec = melspectrogram(y=audio_data)
            z = math.ceil(self.numpy_z_shape_relation * self.duration)
            return np.reshape(mel_spec, (1, 128, z))

        except ValueError as e:
            print("CNN-Feature-Engineer failed:", e)
            return None

    def svc_feature_engineer(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged.
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signals samples with frequency 44100 Hz.
        :return: average of features. (svm working with 1d data of shape (numOfFeatures))
        """

        # TODO: Implement this method
        return
