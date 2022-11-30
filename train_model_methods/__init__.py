"""
@author: Liav Ariel. 2022. All rights reserved.
"""

import librosa
import timeit

__all__ = ['Reader']


class Reader:
    """
    Description: This class is used to read the audio file for training set.
    :param: file_name: Name of the audio file.
    :return: Data of the audio file in the form of numpy array.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        """
        Description: This method is used to read the audio file.
        :return:
        * audio_data as numpy.ndarray:
                A 2-D NumPy array is returned, where the channels are stored
                along the first dimension, i.e. as columns.
                if the sound file has only one channel, a 1-D array is returned.
        * sample_rate as int:
                The sample rate of the audio file [Hz].
        """

        print(f'Reading file: {self.file_name} ...')

        start = timeit.default_timer()

        audio_data, sample_rate = librosa.load(self.file_name, sr=44100, mono=True, duration=5.0)

        stop = timeit.default_timer()

        print(f'File read in {stop - start} seconds.')

        return audio_data, sample_rate
