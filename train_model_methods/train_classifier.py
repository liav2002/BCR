"""
@author: Liav Ariel. 2022. All rights reserved.
"""

from sklearn.preprocessing import \
    StandardScaler  # Standardize features by removing the mean and scaling to unit variance.
from sklearn.svm import SVC  # Support Vector Machine Classifier.
from sklearn.pipeline import Pipeline  # Pipeline of transforms with a final estimator.
from sklearn.model_selection import train_test_split, \
    GridSearchCV  # train_test_split splits data into train set and test set, GridSearchCV performs hyperparameter tuning.
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score  # Metrics for model evaluation.
import numpy as np
import timeit

__all__ = ['TrainClassifier']


class TrainClassifier:
    """
    Description: class to train a classifier of audio_samples.
    """

    def __init__(self, x, y):
        """
        Description: The constructor of the class.
        :param x: inputs.
        :param y: labels.
        """

        self.x = x
        self.y = y

    def train_classifier(self):
        """
        Description: Train random forest classifier.
        :return: pipeline, best_param, best_estimator, perf.
        """

        print('Splitting train and test set. Test set size: 0.25%')

        # Split into training and test set
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=0.25,
                                                            random_state=0,
                                                            stratify=self.y)

        print(f'Train set size: {y_train.size}. Test set size: {y_test.size}.')

        # Create pipeline
        pipeline = Pipeline([
            ('scl', StandardScaler()),  # normalization
            ('clf', SVC(probability=True))
        ])

        # GridSearch
        param_grid = [{'clf__kernel': ['linear', 'rbf'],
                       'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                       'clf__gamma': np.logspace(-2, 2, 5),
                       }]

        # Create grid search object
        estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

        # Fit on data - train the model

        print('Training classifier ...')

        start = timeit.default_timer()

        model = estimator.fit(x_train, y_train)

        stop = timeit.default_timer()

        print(f'Training time: {stop - start} seconds.')

        # Predict

        y_pred = model.predict(x_test)

        perf = {'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'f1': f1_score(y_test, y_pred, average='macro'),
                }

        return perf, model.best_params_, model.best_estimator_
