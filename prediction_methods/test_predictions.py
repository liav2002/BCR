import argparse
import os
import re
import sys
import timeit

import numpy as np
from matplotlib import pyplot as plt

from prediction_methods.prediction import cry_detection

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_PATH)


def test_cnn_predictor():
    parser = argparse.ArgumentParser(description='test cnn model.')
    parser.add_argument('--load_path', default=(PROJECT_PATH + '\\data_for_test_predictions\\'))
    parser.add_argument('--save_path', default=(PROJECT_PATH + '\\output\\test_predictions_result\\'))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # list load_path sub-folders
    directory_list = os.listdir(load_path)

    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative

    print("Testing the cnn model...\n")

    start = timeit.default_timer()

    # iteration on sub-folders
    for directory in directory_list:
        # list files in sub-folder
        file_list = os.listdir(os.path.join(load_path, directory))
        is_baby_cry = directory == 'Crying Baby'

        # iteration on audio files in each sub-folder
        for file in file_list:
            print("try to predict the label of file: ", file)
            # predict the label of the audio file
            result = cry_detection(record=os.path.join(load_path, directory, file), model='cnn')

            # calculate the confusion matrix:

            if result == 1 and is_baby_cry:
                print("file: ", file, " --> TP")
                TP += 1

            if result == 0 and is_baby_cry:
                print("file: ", file, " --> FN")
                FN += 1

            if result == 1 and not is_baby_cry:
                print("file: ", file, " --> FP")
                FP += 1

            if result == 0 and not is_baby_cry:
                print("file: ", file, " --> TN")
                TN += 1

    print("\n", end="")

    stop = timeit.default_timer()

    print(f'Testing time: {stop - start} seconds.')

    # plot the confusion matrix:
    create_confusion_matrix(TP, FP, TN, FN, os.path.join(save_path, "cnn_confusion_matrix.png"),
                            title='CNN - Confusion Matrix')


def test_svc_predictor_with_all_threshold():
    parser = argparse.ArgumentParser(description='test svc model.')
    parser.add_argument('--load_path', default=(PROJECT_PATH + '\\data_for_test_predictions\\'))
    parser.add_argument('--save_path', default=(PROJECT_PATH + '\\output\\test_predictions_result\\'))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # list load_path sub-folders
    directory_list = os.listdir(load_path)

    TPR = []
    FPR = []

    print("Testing the svc model...\n")

    start = timeit.default_timer()

    for threshold in range(1, 11):
        TP = 0  # True Positive
        TN = 0  # True Negative
        FP = 0  # False Positive
        FN = 0  # False Negative

        # iteration on sub-folders
        for directory in directory_list:
            # list files in sub-folder
            file_list = os.listdir(os.path.join(load_path, directory))
            is_baby_cry = directory == 'Crying Baby'

            # iteration on audio files in each sub-folder
            for file in file_list:
                print("try to predict the label of file: ", file)
                # predict the label of the audio file
                result = cry_detection(record=os.path.join(load_path, directory, file), model='svc',
                                       threshold=threshold / 10, num_of_frames=5)

                # calculate the confusion matrix:

                if result == 1 and is_baby_cry:
                    print("file: ", file, " --> TP")
                    TP += 1

                if result == 0 and is_baby_cry:
                    print("file: ", file, " --> FN")
                    FN += 1

                if result == 1 and not is_baby_cry:
                    print("file: ", file, " --> FP")
                    FP += 1

                if result == 0 and not is_baby_cry:
                    print("file: ", file, " --> TN")
                    TN += 1

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        if TP + FN != 0:
            TPR.append(TP / (TP + FN))
        else:
            TPR.append(0)
        if FP + TN != 0:
            FPR.append(FP / (FP + TN))
        else:
            FPR.append(0)

        print("\n", end="")

    stop = timeit.default_timer()

    print(f'Testing time: {stop - start} seconds.')

    # plot the ROC (Receiver Operating Characteristic) curve:
    create_roc_curve(TPR, FPR, os.path.join(save_path, "svc_roc_curve.png"), title='SVC - ROC Curve')


def test_svc_predictor_with_threshold(threshold):
    parser = argparse.ArgumentParser(description='test svc model.')
    parser.add_argument('--load_path', default=(PROJECT_PATH + '\\data_for_test_predictions\\'))
    parser.add_argument('--save_path', default=(PROJECT_PATH + '\\output\\test_predictions_result\\'))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)

    # list load_path sub-folders
    directory_list = os.listdir(load_path)

    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative

    print("Testing the svc model...\n")

    start = timeit.default_timer()

    # iteration on sub-folders
    for directory in directory_list:
        # list files in sub-folder
        file_list = os.listdir(os.path.join(load_path, directory))
        is_baby_cry = directory == 'Crying Baby'

        # iteration on audio files in each sub-folder
        for file in file_list:
            print("try to predict the label of file: ", file)
            # predict the label of the audio file
            result = cry_detection(record=os.path.join(load_path, directory, file), model='svc',
                                   threshold=threshold, num_of_frames=5)

            # calculate the confusion matrix:

            if result == 1 and is_baby_cry:
                print("file: ", file, " --> TP")
                TP += 1

            if result == 0 and is_baby_cry:
                print("file: ", file, " --> FN")
                FN += 1

            if result == 1 and not is_baby_cry:
                print("file: ", file, " --> FP")
                FP += 1

            if result == 0 and not is_baby_cry:
                print("file: ", file, " --> TN")
                TN += 1

    print("\n", end="")

    stop = timeit.default_timer()

    print(f'Testing time: {stop - start} seconds.')

    # plot the confusion matrix:
    create_confusion_matrix(TP, FP, TN, FN, os.path.join(save_path, "svc_confusion_matrix.png"),
                            title="SVC -Confusion Matrix (threshold = " + str(threshold) + ")")


def create_confusion_matrix(TP, FP, TN, FN, save_path, title='Confusion Matrix'):
    x = np.arange(len(["TP", "FP", "TN", "FN"]))
    width = 0.35

    fig, ax = plt.subplots()

    ax.set_ylabel('Number of samples')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(["TP", "FP", "TN", "FN"])

    pps = ax.bar(x - width / 2, [TP, FP, TN, FN], width)

    for p in pps:
        height = p.get_height()
        ax.annotate('{}'.format(height),
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig(save_path)
    plt.close()


def create_roc_curve(TPR, FPR, save_path, title='ROC Curve'):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.plot(FPR, TPR, 'bo')
    plt.plot(FPR, TPR, 'k')
    plt.savefig(save_path)
    plt.close()


def main():
    test_cnn_predictor()
    test_svc_predictor_with_all_threshold()
    test_svc_predictor_with_threshold(0.5)


if __name__ == '__main__':
    main()
