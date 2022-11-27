from project.data_loader import Data


class FeatureSelection():
    digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    face_labels = [0, 1]
    SMALL_VALUE = 0.00000001
    given_data_name_labels = {
        'digit': digit_labels,
        'face': face_labels
    }

    feature_types = ["pixel_wise", "black_white", "grid_black"]

    def __init__(self, train_data_file, train_label_file, train_size, width, height, classifier_data_name,
                 feature_type):

        self.feature_type = feature_type
        self.p_true = {}
        self.p_false = {}

        self.label_count = {}
        self.feature_p_true = {}

        self.train_data_loader = Data(train_data_file, train_label_file, train_size, width, height)

        self.given_data_name_labels = self.given_data_name_labels[classifier_data_name]

        for label in self.given_data_name_labels:
            self.p_true[label] = 0
            self.p_false[label] = 0

        for label in self.train_data_loader.labels:
            self.p_true[label] += 1

        self.label_count = self.p_true.copy()

        for label in self.given_data_name_labels:
            # self.label_count[label] = self.p_true[label]

            self.p_true[label] /= train_size
            self.p_false[label] = 1 - self.p_true[label]

    def create_lookup_features(self):

        if self.feature_type == "black_white":
            pixel_length = self.train_data_loader.width * self.train_data_loader.height

            feature_with_label_black = {}
            feature_with_label_white = {}
            for label in self.label_count:
                pixel_dict = {}
                for pixel in range(pixel_length + 1):
                    pixel_dict[pixel] = 0

                feature_with_label_black[label] = pixel_dict
                feature_with_label_white[label] = {}

            # initialize black features
            for mat_index, matrix in enumerate(self.train_data_loader.matrices):
                label = self.train_data_loader.labels[mat_index]

                black_count = 0
                for i in range(self.train_data_loader.height):
                    for j in range(self.train_data_loader.width):
                        if matrix[i][j] == 1:
                            black_count += 1

                count_map = feature_with_label_black[label]
                count_map[black_count] = count_map.get(black_count, 0) + 1

            # compute white count
            for label in feature_with_label_black:
                label_black_map = feature_with_label_black[label]
                white_label_map = feature_with_label_white[label]
                for pixel in range(pixel_length + 1):
                    white_label_map[pixel] = self.label_count[label] - label_black_map[pixel]

            # Compute the black and white feature true probabilities

            black_prob_true = {}
            white_prob_true = {}

            black_total_sum = {}
            white_total_sum = {}

            for label in self.label_count:
                for i in range(pixel_length + 1):
                    black_count_i = feature_with_label_black[label].get(i, 0)
                    white_count_i = feature_with_label_white[label].get(i, 0)
                    label_count = self.label_count[label]

                    black_total_sum[i] = black_total_sum.get(i, 0) + black_count_i
                    white_total_sum[i] = white_total_sum.get(i, 0) + white_count_i

                    black_p = black_count_i / label_count
                    if black_p == 0:
                        black_p = self.SMALL_VALUE
                    black_prob_true[(i, label)] = black_p

                    white_p = white_count_i / label_count
                    if white_p == 0:
                        white_p = self.SMALL_VALUE
                    white_prob_true[(i, label)] = white_p

            self.black_prob_true = black_prob_true
            self.white_prob_true = white_prob_true

            #           # Compute the black and white feature false probabilities
            black_prob_false = {}
            white_prob_false = {}

            total_label_count = self.train_data_loader.size
            for label in self.label_count:
                for i in range(pixel_length + 1):

                    black_count_i = feature_with_label_black[label].get(i, 0)
                    white_count_i = feature_with_label_white[label].get(i, 0)
                    label_count = self.label_count[label]

                    black_total_sum_diff_label = black_total_sum.get(i) - black_count_i
                    diff_label_count = total_label_count - label_count

                    black_p = black_total_sum_diff_label / diff_label_count
                    if black_p == 0:
                        black_p = self.SMALL_VALUE
                    black_prob_false[(i, label)] = black_p

                    white_total_sum_diff_label = white_total_sum.get(i) - white_count_i

                    white_p = white_total_sum_diff_label / diff_label_count
                    if white_p == 0:
                        white_p = self.SMALL_VALUE
                    white_prob_false[(i, label)] = white_p

            self.black_prob_false = black_prob_false
            self.white_prob_false = white_prob_false
            print('hello')

    def make_prediction(self, matrix):
        if self.feature_type == "black_white":
            black_count = 0
            white_count = 0

            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if matrix[i][j] == 1:
                        black_count += 1
                    else:
                        white_count += 1

            ld = {}
            for label in self.given_data_name_labels:
                # computing likelehood for each
                prob_num = self.p_true[label] * self.black_prob_true[(black_count, label)] * self.white_prob_true[
                    (black_count, label)]
                prob_denom = self.p_false[label] * self.black_prob_false[(white_count, label)] * self.white_prob_false[
                    (white_count, label)]
                ld[label] = prob_num / prob_denom
            output = max(ld, key=ld.get)
            return output


if __name__ == "__main__":
    # train_model = FeatureSelection("/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrain",
    #                                "/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrainlabels",
    #                                400, 60, 70,
    #                                'face', 'black_white')
    #
    # test_data = Data("/Users/pranoysarath/Downloads/classification/data/facedata/facedatatest",
    #                  "/Users/pranoysarath/Downloads/classification/data/facedata/facedatatestlabels",
    #                  150, 60, 70)
    # train_model.create_lookup_features()
    #
    # correct_predictions = 0
    #
    # for i in range(150):
    #
    #     output = train_model.make_prediction(test_data.matrices[i])
    #     if output == test_data.labels[i]:
    #         correct_predictions += 1
    #
    # print(correct_predictions * 100 / 150)

    train_model = FeatureSelection("/Users/pranoysarath/Downloads/classification/data/digitdata/trainingimages",
                                   "/Users/pranoysarath/Downloads/classification/data/digitdata/traininglabels", 400,
                                   28, 28,
                                   'digit', 'black_white')

    train_model.create_lookup_features()
    correct = 0
    for i in range(100):
        output = train_model.make_prediction(train_model.train_data_loader.matrices[i])
        if output == train_model.train_data_loader.labels[i]:
            correct += 1
    print(correct)
