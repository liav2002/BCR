import argparse
import os
import sys

from tensorflow import keras

from prediction_methods import Reader
from prediction_methods.feature_engineer import FeatureEngineer
from prediction_methods.majority_voter import MajorityVoter
from prediction_methods.baby_cry_predictor import BabyCryPredictor

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_PATH)


def cnn_cry_detection(record, threshold=0.5, record_second_length=5, num_of_frames=1):
    """
    Description: This method is used to predict the audio file using CNN model.
    :param record_second_length: the length of the audio file in seconds.
    :param num_of_frames: number of frames to divide the audio file.
    :param threshold: confidence threshold.
    :param record: file - full path.
    :return: 1 if the audio file is a baby cry, 0 otherwise.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_model',
                        default=(PROJECT_PATH + '\\output\\model\\'))
    parser.add_argument('--file_name', default=record)

    # Arguments
    args = parser.parse_args()
    load_path_model = os.path.normpath(args.load_path_model)
    file_name = args.file_name

    # READ RAW SIGNAL (first 2 sec)
    file_reader = Reader(file_name=file_name, num_of_frames=num_of_frames, record_seconds_length=record_second_length)
    play_list = file_reader.read_audio_file()

    # FEATURE ENGINEERING
    engineer = FeatureEngineer(duration=record_second_length / num_of_frames)
    play_list_processed = list()
    for signal in play_list:
        tmp = engineer.cnn_feature_engineer(signal)
        play_list_processed.append(tmp)

    # OPEN MODEL
    model = keras.models.load_model(os.path.join(load_path_model, 'cnn\\'))

    # PREDICT
    predictor = BabyCryPredictor(model)
    predictions = list()

    for mel_spec in play_list_processed:
        tmp = predictor.cnn_classify(mel_spec)
        predictions.append(tmp)

    # MAJORITY VOTER
    majority_voter = MajorityVoter(predictions, threshold)
    majority_vote = majority_voter.vote()

    return majority_vote


def svc_cty_detection(record):
    # TODO: implement
    return


def multi_model_detection(record):
    # TODO: implement
    return
