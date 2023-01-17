import argparse
import os
import pickle
import sys
import warnings

from tensorflow import keras

from prediction_methods import Reader
from prediction_methods.feature_engineer import FeatureEngineer
from prediction_methods.majority_voter import MajorityVoter
from prediction_methods.baby_cry_predictor import BabyCryPredictor

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_PATH)


def cry_detection(record, model, threshold=0.5, record_second_length=5, num_of_frames=1):
    """
    Description: This method is used to predict the audio file using CNN model.
    :param record_second_length: the length of the audio file in seconds.
    :param model: the model to use for prediction.
    :param num_of_frames: number of frames to divide the audio file.
    :param threshold: confidence threshold.
    :param record: file - full path.
    :return: 1 if the audio file is a baby cry, 0 otherwise.
    """

    # Check the model type
    if model != 'cnn' and model != 'svc':
        raise ValueError('The model type is not supported.')

    # TODO: try figure out how to use out cnn mode with audio file that is not 5 seconds long, until that:
    if model == 'cnn' and num_of_frames != 1:
        num_of_frames = 1
    if model == 'cnn' and record_second_length != 5:
        print("Warning: the audio file is not 5 seconds long, maybe the prediction will throw an error.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_model',
                        default=(PROJECT_PATH + '\\output\\model\\'))
    parser.add_argument('--file_name', default=record)

    # Arguments
    args = parser.parse_args()
    load_path_model = os.path.normpath(args.load_path_model)
    file_name = args.file_name

    # READ AUDIO FILE, SPLIT TO FRAMES AND EXTRACT FEATURES
    file_reader = Reader(file_name=file_name, num_of_frames=num_of_frames, record_seconds_length=record_second_length)
    play_list = file_reader.read_audio_file()

    # FEATURE ENGINEERING
    engineer = FeatureEngineer(duration=record_second_length / num_of_frames)
    play_list_processed = list()
    for signal in play_list:
        tmp = engineer.cnn_feature_engineer(signal) if model == 'cnn' else engineer.svc_feature_engineer(signal)
        play_list_processed.append(tmp)

    cnn_model = None
    svc_model = None

    # OPEN MODEL
    if model == 'cnn':
        cnn_model = keras.models.load_model(os.path.join(load_path_model, 'cnn\\'))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open((os.path.join(load_path_model, 'svc_best_model.pkl')), 'rb') as fp:
                svc_model = pickle.load(fp)

    # PREDICT
    predictions = list()

    if cnn_model is not None:
        cnn_predictor = BabyCryPredictor(cnn_model)
        for mel_spec in play_list_processed:
            tmp = cnn_predictor.cnn_classify(mel_spec)
            predictions.append(tmp)

    elif svc_model is not None:
        svc_predictor = BabyCryPredictor(svc_model)
        for signal in play_list_processed:
            tmp = svc_predictor.svc_classify(signal)
            predictions.append(tmp)

    else:
        raise ValueError('Failed to load model.')

    # MAJORITY VOTER
    majority_voter = MajorityVoter(predictions, threshold)
    majority_vote = majority_voter.vote()

    return majority_vote
