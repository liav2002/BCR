"""
@author: Liav Ariel. 2022. All rights reserved.
"""

from train_model_methods import Reader
from train_model_methods.cnn_feature_engineer import CNNFeatureEngineer
import os
import re
import numpy as np
import argparse
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_PATH)

label_to_int_dict = {
    "301 - Crying baby": 0,
    "901 - Silence": 1,
    "902 - Noise": 2,
    "903 - Baby laugh": 3
}


def main():
    """
    Description: This method is used to read the audio file and engineer the feature.
                 The result is the dataset for training the model.
    :return: None
    """

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Read the audio file and engineer the feature.')
    parser.add_argument('--load_path', default=(PROJECT_PATH + '\\data\\'))
    parser.add_argument('--save_path', default=(PROJECT_PATH + '\\output\\dataset\\'))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # list load_path sub-folders
    regex = re.compile(r'^[0-9]')
    directory_list = [i for i in os.listdir(load_path) if regex.search(i)]

    # initialize empty array for features
    x = np.zeros([1, 128, 431])

    # initialize empty array for labels
    y = np.zeros(0)

    print("Creating dataset for training the model...")

    # iteration on sub-folders
    for directory in directory_list:
        # Instantiate FeatureEngineer
        cnn_feature_engineer = CNNFeatureEngineer(label=directory)

        # list files in sub-folder
        file_list = os.listdir(os.path.join(load_path, directory))

        # iteration on audio files in each sub-folder
        for file in file_list:
            # Instantiate Reader
            reader = Reader(file_name=os.path.join(load_path, directory, file))

            # Read audio file
            audio_data, sample_rate = reader.read_audio_file()

            # Engineer features
            mel_spec, label = cnn_feature_engineer.feature_engineer(audio_data=audio_data)

            # Append feature to x (dimensionality is 1, size is 128*431)
            x = np.append(x, mel_spec)

            # Append label to y
            integer = label_to_int_dict[label]
            y = np.append(y, [integer], axis=0)

    # Delete first row of x
    x = x.reshape(567, 128, 431)
    x = np.delete(x, 0, axis=0)

    # save dataset
    print("Saving training dataset...")
    np.save(os.path.join(save_path, 'cnn_dataset.npy'), x)
    np.save(os.path.join(save_path, 'cnn_labels.npy'), y)

    print(f"Saved! {os.path.join(save_path, 'cnn_dataset.npy')}")
    print(f"Saved! {os.path.join(save_path, 'cnn_labels.npy')}")


if __name__ == '__main__':
    main()
