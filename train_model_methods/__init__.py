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
        """
        print('Reading data...')
        # TODO: Read the audio file.
