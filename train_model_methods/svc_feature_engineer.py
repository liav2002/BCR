"""
@author: Liav Ariel. 2022. All rights reserved.
"""

import numpy as np
import timeit
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, \
    chroma_cens, rms

__all__ = ['FeatureEngineer']


class FeatureEngineer:
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
            zcr_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='zero_crossing_rate')
            rmse_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='rmse')
            mfcc_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='mfcc')
            spectral_centroid_feature = self.compute_librosa_features(audio_data=audio_data,
                                                                      feature_name='spectral_centroid')
            spectral_rolloff_feature = self.compute_librosa_features(audio_data=audio_data,
                                                                     feature_name='spectral_rolloff')
            spectral_bandwidth_feature = self.compute_librosa_features(audio_data=audio_data,
                                                                       feature_name='spectral_bandwidth')

            concat_features = np.concatenate((zcr_feature,
                                              rmse_feature,
                                              mfcc_feature,
                                              spectral_centroid_feature,
                                              spectral_rolloff_feature,
                                              spectral_bandwidth_feature
                                              ), axis=0)

            print("Averaging features...")

            start = timeit.default_timer()

            mean_features = np.mean(concat_features, axis=1, keepdims=True).transpose()

            stop = timeit.default_timer()

            print(f'Features averaged in {stop - start} seconds.')

            return mean_features, self.label

        except ValueError as e:
            print(e)
            return None, None

    def compute_librosa_features(self, audio_data, feature_name):
        """
        Description: This method is used to compute feature using librosa methods.
        :param: audio_data: The input signals samples with frequency 44100 Hz.
        :param: feature_name: The name of the feature to compute.
        :return: a numpy array (numOfFeatures x numOfShortTermWindows).
        """

        print(f"Computing {feature_name}...")

        if feature_name == 'zero_crossing_rate':
            return zero_crossing_rate(y=audio_data, hop_length=self.FRAME)
        elif feature_name == 'rmse':
            return rms(y=audio_data, hop_length=self.FRAME)
        elif feature_name == 'mfcc':
            return mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)
        elif feature_name == 'spectral_centroid':
            return spectral_centroid(y=audio_data, sr=self.RATE, hop_length=self.FRAME)
        elif feature_name == 'spectral_rolloff':
            return spectral_rolloff(y=audio_data, sr=self.RATE, hop_length=self.FRAME, roll_percent=0.90)
        elif feature_name == 'spectral_bandwidth':
            return spectral_bandwidth(y=audio_data, sr=self.RATE, hop_length=self.FRAME)
        else:
            raise ValueError(f"ERROR: feature name {feature_name} is not supported.")
