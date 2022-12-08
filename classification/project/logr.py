"""
Uses Scikit Logistic Regression library to build a classifier model

"""

import time

from classification.project.data_loader import Data
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import numpy as np

from classification.project.datapath import DataPath


class LogR():
    digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    face_labels = [0, 1]
    given_data_name_labels = {
        'digit': digit_labels,
        'face': face_labels
    }

    train_data_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_sampling_iteration = 5

    def __init__(self, train_data_file, train_label_file, train_size, width, height, classifier_data_name,
                 feature_type):
        self.feature_type = feature_type
        self.label_count = {}
        self.feature_p_true = {}

        self.train_data_loader = Data(train_data_file, train_label_file, train_size, width, height)

        self.given_data_name_labels = self.given_data_name_labels[classifier_data_name]
        self.pixel_count = width * height

        self.full_data = None
        self.data = None

    def get_numpy_array_from_matrix(self, matrix):
        data_point = [1]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                data_point.append(matrix[i][j])
        return np.array(data_point)

    def create_feature(self, train_size):
        if self.full_data is None:
            data = []
            for matrix in self.train_data_loader.matrices:
                data.append(self.get_numpy_array_from_matrix(matrix))
            self.full_data = data

        if train_size == 1:
            self.data = self.full_data
            self.train_sample_labels = self.train_data_loader.labels
        else:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.full_data,
                                                                                        self.train_data_loader.labels,
                                                                                        test_size=1 - train_size)
            self.train_sample_labels = y_train
            self.data = X_train

    def train_model(self):
        # neigh = KNeighborsClassifier(n_neighbors=10)
        # neigh.fit(self.data, self.train_data_loader.labels)
        # self.knn_model = neigh

        start = time.time()
        lR = LogisticRegression(random_state=0)
        lR.fit(self.data, self.train_sample_labels)
        self.lR = lR
        end = time.time()
        return end - start

    def make_prediction(self, matrix):
        # return self.knn_model.predict([self.get_numpy_array_from_matrix(matrix)])
        return self.lR.predict([self.get_numpy_array_from_matrix(matrix)])

    @staticmethod
    def face_training_prediction():
        face_model = LogR(DataPath.getPath(0, 'TRAINING_FACE_DATA_PATH'),
                          DataPath.getPath(0, 'TRAINING_FACE_LABEL_PATH'),
                          400, 60, 70,
                          'face', 'pixel')

        face_test_data = Data(DataPath.getPath(0, 'TEST_FACE_DATA_PATH'),
                              DataPath.getPath(0, 'TEST_FACE_LABEL_PATH'),
                              150, 60, 70)

        accuracy = {}
        for i in LogR.train_data_sizes:
            accuracy[i] = []
        for train_size in LogR.train_data_sizes:
            for iteration in range(LogR.train_sampling_iteration):
                face_model.create_feature(train_size)
                train_time = face_model.train_model()
                correct_predictions = 0
                for i in range(len(face_test_data.matrices)):
                    output = face_model.make_prediction(face_test_data.matrices[i])
                    if output == face_test_data.labels[i]:
                        correct_predictions += 1
                prediction_accuracy = correct_predictions / len(face_test_data.matrices)
                accuracy[train_size].append((prediction_accuracy, train_time))
                print(
                    f"Accuracy for face with train size: {train_size * 100} with iteration {iteration + 1} is {prediction_accuracy * 100}")
        print(f"Accuracy for face with train size: {accuracy}")

    @staticmethod
    def digit_training_prediction():
        digit_model = LogR(DataPath.getPath(0, 'TRAINING_DIGIT_DATA_PATH'),
                           DataPath.getPath(0, 'TRAINING_DIGIT_LABEL_PATH'), 5000,
                           28, 28,
                           'digit', 'pixel')
        digit_test_data = Data(DataPath.getPath(0, 'TEST_DIGIT_DATA_PATH'),
                               DataPath.getPath(0, 'TEST_DIGIT_LABEL_PATH'), 1000,
                               28, 28, )
        accuracy = {}
        for i in LogR.train_data_sizes:
            accuracy[i] = []
        for train_size in LogR.train_data_sizes:
            for iteration in range(LogR.train_sampling_iteration):
                digit_model.create_feature(train_size)
                train_time = digit_model.train_model()
                correct_predictions = 0
                for i in range(len(digit_test_data.matrices)):
                    output = digit_model.make_prediction(digit_test_data.matrices[i])
                    if output == digit_test_data.labels[i]:
                        correct_predictions += 1
                prediction_accuracy = correct_predictions / len(digit_test_data.matrices)
                accuracy[train_size].append((prediction_accuracy, train_time))
                print(
                    f"Accuracy for digit with train size: {train_size * 100} with iteration {iteration + 1} is {prediction_accuracy * 100}")
        print(f"Accuracy for digit with train size: {accuracy}")


if __name__ == "__main__":
    LogR.face_training_prediction()
    LogR.digit_training_prediction()
