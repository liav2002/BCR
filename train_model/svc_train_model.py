"""
@author: Liav Ariel. 2022. All rights reserved.
"""

import argparse
import json
import os
import pickle
import numpy as np

from train_model_methods.svc_train_classifier import TrainClassifier

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


def main():
    """
    Description: This method is used to train the model.
    :return: None
    """

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--load_path', default=(PROJECT_PATH + '\\output\\dataset\\'))
    parser.add_argument('--save_path', default=(PROJECT_PATH + '\\output\\model\\'))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # Load dataset
    x = np.load(os.path.join(load_path, 'dataset.npy'))
    y = np.load(os.path.join(load_path, 'labels.npy'))

    # Instantiate TrainClassifier
    train_classifier = TrainClassifier(x=x, y=y)

    # Train classifier
    performance, parameters, best_estimator = train_classifier.train_classifier()

    # Save model
    print("Saving the model...")

    # Save model performance
    with open(os.path.join(save_path, 'model_performance.json'), 'w') as fp:
        json.dump(performance, fp)

    # Save best parameters
    with open(os.path.join(save_path, 'best_parameters.json'), 'w') as fp:
        json.dump(parameters, fp)

    # Save best estimator
    with open(os.path.join(save_path, 'best_estimator.pkl'), 'wb') as fp:
        pickle.dump(best_estimator, fp)

    print(f"Saved! {os.path.join(save_path, 'model_performance.json')}")
    print(f"Saved! {os.path.join(save_path, 'best_parameters.json')}")
    print(f"Saved! {os.path.join(save_path, 'best_estimator.pkl')}")


if __name__ == '__main__':
    main()
