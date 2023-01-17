import numpy as np
import math

__all__ = ['FeatureEngineer']

from librosa.feature import melspectrogram
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms


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

        zcr_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='zero_crossing_rate')
        rmse_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='rmse')
        mfcc_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='mfcc')
        spectral_centroid_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='spectral_centroid')
        spectral_rolloff_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='spectral_rolloff')
        spectral_bandwidth_feat = self.compute_librosa_features(audio_data=audio_data, feat_name='spectral_bandwidth')

        concat_feat = np.concatenate((zcr_feat,
                                      rmse_feat,
                                      mfcc_feat,
                                      spectral_centroid_feat,
                                      spectral_rolloff_feat,
                                      spectral_bandwidth_feat
                                      ), axis=0)

        return np.mean(concat_feat, axis=1, keepdims=True).transpose()

    def compute_librosa_features(self, audio_data, feat_name):
        """
        Compute feature using librosa methods

        :param audio_data: signal
        :param feat_name: feature to compute
        :return: np array
        """

        if feat_name == 'zero_crossing_rate':
            return zero_crossing_rate(y=audio_data, hop_length=self.frame)
        elif feat_name == 'rmse':
            return rms(y=audio_data, hop_length=self.frame)
        elif feat_name == 'mfcc':
            return mfcc(y=audio_data, sr=self.rate, n_mfcc=13)
        elif feat_name == 'spectral_centroid':
            return spectral_centroid(y=audio_data, sr=self.rate, hop_length=self.frame)
        elif feat_name == 'spectral_rolloff':
            return spectral_rolloff(y=audio_data, sr=self.rate, hop_length=self.frame, roll_percent=0.90)
        elif feat_name == 'spectral_bandwidth':
            return spectral_bandwidth(y=audio_data, sr=self.rate, hop_length=self.frame)
