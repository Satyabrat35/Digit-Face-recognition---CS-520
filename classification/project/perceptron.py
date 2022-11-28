from classification.project.data_loader import Data
import numpy as np


class Perceptron():
    digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    face_labels = [0, 1]
    given_data_name_labels = {
        'digit': digit_labels,
        'face': face_labels
    }

    learning_rate = 0.001
    iterations = 30

    def __init__(self, train_data_file, train_label_file, train_size, width, height, classifier_data_name,
                 feature_type):
        self.feature_type = feature_type
        self.label_count = {}
        self.feature_p_true = {}

        self.train_data_loader = Data(train_data_file, train_label_file, train_size, width, height)

        self.given_data_name_labels = self.given_data_name_labels[classifier_data_name]
        self.pixel_count = width * height

    def get_numpy_array_from_matrix(self, matrix):
        data_point = [1]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                data_point.append(matrix[i][j])
        return np.array(data_point)

    def create_feature(self):

        data = []
        for matrix in self.train_data_loader.matrices:
            data.append(self.get_numpy_array_from_matrix(matrix))

        self.data = data

    def update_weight_vector(self, weight_vector, input_vector, increment=False):
        if increment:
            weight_vector[0] += 1
        else:
            weight_vector[0] -= 1

        for i in range(1, len(weight_vector)):
            if increment:
                weight_vector[i] += self.learning_rate * input_vector[i]
            else:
                weight_vector[i] -= self.learning_rate * input_vector[i]

    def train_model(self):

        label_size = len(self.given_data_name_labels)
        actual_labels = self.train_data_loader.labels

        # matrix init
        weights = np.zeros((label_size, self.pixel_count + 1))

        for iteration in range(self.iterations):
            print(f"Epoch {iteration}")
            change = False
            for index, data_point in enumerate(self.data):
                result = np.matmul(data_point, np.transpose(weights))
                predicted_label = np.argmax(result)

                if predicted_label != actual_labels[index]:
                    #                     decrement predicted_label
                    self.update_weight_vector(weights[predicted_label], data_point, False)
                    #                     increment actual label
                    self.update_weight_vector(weights[actual_labels[index]], data_point, True)
                    change = True
            if not change:
                break
        self.weights = weights

    def make_prediction(self, matrix):
        data = self.get_numpy_array_from_matrix(matrix)
        function_output = np.matmul(data, np.transpose(self.weights))
        return np.argmax(function_output)


if __name__ == "__main__":
    face_model = Perceptron("/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrain",
                            "/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrainlabels",
                            400, 60, 70,
                            'face', 'pixel')

    face_test_data = Data("/Users/pranoysarath/Downloads/classification/data/facedata/facedatatest",
                          "/Users/pranoysarath/Downloads/classification/data/facedata/facedatatestlabels",
                          150, 60, 70)
    face_model.create_feature()
    face_model.train_model()

    correct_predictions = 0
    for i in range(len(face_test_data.matrices)):
        output = face_model.make_prediction(face_test_data.matrices[i])
        if output == face_test_data.labels[i]:
            correct_predictions += 1
    print(f"Accuracy for face is {correct_predictions * 100 / len(face_test_data.matrices)}")

    digit_model = Perceptron("/Users/pranoysarath/Downloads/classification/data/digitdata/trainingimages",
                             "/Users/pranoysarath/Downloads/classification/data/digitdata/traininglabels", 5000,
                             28, 28,
                             'digit', 'pixel')
    digit_test_data = Data("/Users/pranoysarath/Downloads/classification/data/digitdata/testimages",
                           "/Users/pranoysarath/Downloads/classification/data/digitdata/testlabels", 1000,
                           28, 28, )
    digit_model.create_feature()
    digit_model.train_model()
    correct_predictions = 0
    for i in range(len(digit_test_data.matrices)):
        output = digit_model.make_prediction(digit_test_data.matrices[i])
        if output == digit_test_data.labels[i]:
            correct_predictions += 1
    print(f"Accuracy for digit is {correct_predictions * 100 / len(digit_test_data.matrices)}")
