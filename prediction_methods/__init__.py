import librosa

__all__ = ['Reader']


class Reader:
    """
    Read audio file. Divide him into frames. and Return list of data frames.
    file_name: 'path/to/file.mp3'
    """

    def __init__(self, file_name, record_seconds_length=5, num_of_frames=5):
        self.file_name = file_name
        self.num_of_frames = num_of_frames
        self.duration = record_seconds_length / num_of_frames

    def read_audio_file(self):
        """
        Read audio file. Divide him into frames. and Return list of data frames.
        """

        play_list = list()

        for offset in range(0, self.num_of_frames):
            audio_data, sample_rate = librosa.load(self.file_name, sr=44100, mono=True, offset=offset,
                                                   duration=self.duration)
            play_list.append(audio_data)

        return play_list
