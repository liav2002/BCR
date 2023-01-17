"""
@author: Liav Ariel. 2022. All rights reserved.
"""

import os
import re
import numpy as np
import argparse

from train_model_methods import Reader
from train_model_methods.svc_feature_engineer import FeatureEngineer

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


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
    x = np.empty([1, 18])

    # initialize empty array for labels
    y = []

    print("Creating dataset for training the model...")

    # iteration on sub-folders
    for directory in directory_list:
        # Instantiate FeatureEngineer
        feature_engineer = FeatureEngineer(label=directory)

        # list files in sub-folder
        file_list = os.listdir(os.path.join(load_path, directory))

        # iteration on audio files in each sub-folder
        for file in file_list:
            # Instantiate Reader
            reader = Reader(file_name=os.path.join(load_path, directory, file))

            # Read audio file
            audio_data, sample_rate = reader.read_audio_file()

            # Engineer features
            avg_features, label = feature_engineer.feature_engineer(audio_data=audio_data)

            # Append feature to x
            x = np.concatenate((x, avg_features), axis=0)

            # Append label to y
            y.append(label)

    # Delete first row of x
    x = np.delete(x, 0, 0)

    # save dataset
    print("Saving training dataset...")
    np.save(os.path.join(save_path, 'svc_dataset.npy'), x)
    np.save(os.path.join(save_path, 'svc_labels.npy'), y)

    print(f"Saved! {os.path.join(save_path, 'svc_dataset.npy')}")
    print(f"Saved! {os.path.join(save_path, 'svc_labels.npy')}")


if __name__ == '__main__':
    main()
