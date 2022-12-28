"""
@author: Jonah Hess. 2022. All rights reserved.
"""

from keras import layers, models
from sklearn.model_selection import train_test_split
import timeit


class CNNTrainClassifier:
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
        Description: Train CNN-Based classifier.
        :return: perf, history.
        """

        print('Splitting train and test set. Test set size: 0.33%')

        # Split into training and test set
        # note: in regular classifier, random_state = 0, stratify = self.y
        X_train, X_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=0.33,
                                                            random_state=42)

        X_val, X_test, y_val, y_test = train_test_split(X_test,
                                                        y_test,
                                                        test_size=0.5,
                                                        random_state=42)

        print(f'Train set size: {y_train.size}. Test set size: {y_test.size}.')

        # Create CNN Model
        # todo: initialize model using initializer
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 431, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='softmax'))
        model.add(layers.Dense(4))

        model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
        model.summary()

        start = timeit.default_timer()

        print('Training classifier ...')

        # Fit on data - train the model
        history = model.fit(X_train, y_train, epochs=3,
                            validation_data=(X_val, y_val))

        stop = timeit.default_timer()

        print(f'Training time: {stop - start} seconds.')

        # Predict
        score = model.evaluate(X_test, y_test, verbose=2)
        print(score)

        perf = {}

        return perf, history
