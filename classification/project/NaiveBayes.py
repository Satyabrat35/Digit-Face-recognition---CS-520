import pandas as pd

from classification.project.data_loader import Data
from classification.project.datapath import DataPath
import numpy as np
import math
from sklearn.metrics import accuracy_score
import random
import statistics
import matplotlib.pyplot as plt
import copy
from timeit import default_timer as timer


class NaiveBayes():
    digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    face_labels = [0, 1]
    given_data_type_labels = {'digit': digit_labels, 'face': face_labels}
    given_data_name_labels = None
    train_information_matrix = []
    true_label_prob = []
    feature_matrix = []
    matrix_element_count = -1
    actual_labels = []

    def __init__(self, train_data_file, train_label_file, train_size, width, height, classifier_data_name,
                 feature_selection):
        self.train_data_loader = Data(train_data_file, train_label_file, train_size, width, height)
        self.given_data_name_labels = self.given_data_type_labels[classifier_data_name]
        self.actual_labels = self.train_data_loader.labels
        arg_s = self.get_arg_s_for_feature_selection(feature_selection, self.train_data_loader.matrices,
                                                     self.actual_labels)

        # if feature_selection == 'pixel' or feature_selection == 'black_count':
        #     arg_s = [self.train_data_loader.matrices, self.actual_labels , True ]
        #     self.matrix_element_count = len(self.train_data_loader.matrices[0]) * len(self.train_data_loader.matrices[0][0])
        # else:
        #     arg_s = [self.train_data_loader.matrices, 4, 4, 2]  #matrices, g_length, g_wisth , jump_values
        #     self.matrix_element_count =  int((((len(self.train_data_loader.matrices[0]) - arg_s[1])/arg_s[3]) +1)* (((len(self.train_data_loader.matrices[0][0]) - arg_s[2])/arg_s[3]) + 1))

        self.feature_matrix = self.get_feature_selection(feature_selection, arg_s, self.actual_labels)
        # self.true_label_prob = self.trueLabelProbability(self.actual_labels, self.given_data_name_labels )

    def get_arg_s_for_feature_selection(self, feature_selection, data_matrix, actual_labels=None, process='train'):
        if feature_selection == 'pixel' or feature_selection == 'black_count':
            if process == 'train':
                arg_s = [data_matrix, actual_labels, True]
            elif process == 'test':
                arg_s = [data_matrix, actual_labels, False]
            self.matrix_element_count = len(data_matrix[0]) * len(data_matrix[0][0])
        else:
            arg_s = [data_matrix, 2, 2, 2]
            # if(classifier_data_name == 'digit'):
            #     arg_s = [data_matrix, 4, 4, 2]  # matrices, g_length, g_wisth , jump_values
            # elif(classifier_data_name == 'face'):
            #     arg_s = [data_matrix, 4, 4, 2]
            self.matrix_element_count = int((((len(data_matrix[0]) - arg_s[1]) / arg_s[3]) + 1) * (
                    ((len(data_matrix[0][0]) - arg_s[2]) / arg_s[3]) + 1))
        return arg_s

    def train_naive_model(self, feature_selection, k=2, training_ratio=1.0):
        training_count = len(self.actual_labels)
        random_training_ids = random.sample(range(training_count), int(training_ratio * training_count))
        self.feature_matrix = list(map(self.feature_matrix.__getitem__, random_training_ids))
        self.actual_labels = list(map(self.actual_labels.__getitem__, random_training_ids))
        self.true_label_prob = self.trueLabelProbability(self.actual_labels, self.given_data_name_labels)
        if feature_selection == 'black_count':
            self.train_information_matrix = self.probablity_of_fi_given_y_bp(pd.DataFrame(self.feature_matrix),
                                                                             self.given_data_name_labels,
                                                                             self.matrix_element_count, k, False)
        elif feature_selection == 'pixel':
            self.train_information_matrix = self.probability_of_fi_given_y_px(pd.DataFrame(self.feature_matrix),
                                                                              self.given_data_name_labels, k, True)

    def get_feature_selection(self, feature_selection, args, actual_labels=None, process='train'):
        if feature_selection == 'pixel':
            return Data.pixelFeatureSet(self, args[0], args[1], args[2])
        elif feature_selection == 'black_count':
            return Data.blackPixelFeatureSet(self, args[0], args[1], args[2])
        elif feature_selection == 'reduced_black_count':
            args[0] = Data.reducedGridFeatureSet(self, args[0], args[1], args[2], args[3], True)
            args = self.get_arg_s_for_feature_selection('black_count', args[0], actual_labels, process)
            return Data.blackPixelFeatureSet(self, args[0], args[1], args[2])
        elif feature_selection == 'reduced_pixel':
            args[0] = Data.reducedGridFeatureSet(self, args[0], args[1], args[2], args[3], True)
            args = self.get_arg_s_for_feature_selection('pixel', args[0], actual_labels, process)
            return Data.pixelFeatureSet(self, args[0], args[1], args[2])
        elif feature_selection == 'reduced_grid':
            return Data.reducedGridFeatureSet(self, args[0], args[1], args[2], args[3])

    def trueLabelProbability(self, actual_labels, given_data_name_labels):  # code testted
        label_dict = {key: 0 for key in given_data_name_labels}
        for x in actual_labels:
            label_dict[x] += 1
        true_label_prob = np.array(list(label_dict.values())) / len(actual_labels)
        return true_label_prob

    def evaluation_metric(slef, actual_labels, predicted_labels):
        # use some libraray to calculate these metrics
        return accuracy_score(actual_labels, predicted_labels)

    """Below code is specific to black and white pixel information only"""
    # k is smoothing factor
    import math
    def probablity_of_fi_given_y_bp(self, training_bwp_data_df, given_data_name_labels, matrix_element_count, k,
                                    need_log=False):
        prob_of_fi_given_y = []
        for var_y in given_data_name_labels:
            training_data_wrt_y = training_bwp_data_df[training_bwp_data_df.iloc[:, -1].values == var_y].values
            blackp_counter_dict, whitep_counter_dict = {}, {}
            for data_point in training_data_wrt_y:
                blackp_counter_dict[data_point[0]] = blackp_counter_dict.get(data_point[0], 0) + 1
                whitep_counter_dict[data_point[1]] = whitep_counter_dict.get(data_point[1], 0) + 1
            prob_blackp_counter_list, prob_whitep_counter_list = [], []
            for i in range(matrix_element_count):
                no_information_value = round(k / (len(training_data_wrt_y) + len(given_data_name_labels) * k), 2)
                if need_log == True:
                    no_information_value = round(math.log(no_information_value), 2)

                if i not in blackp_counter_dict.keys():
                    prob_blackp_counter_list.append(no_information_value)
                else:
                    informative_value = round(
                        (blackp_counter_dict[i] + k) / (len(training_data_wrt_y) + len(given_data_name_labels) * k), 2)
                    if need_log == True:
                        informative_value = round(math.log(informative_value), 2)
                    prob_blackp_counter_list.append(informative_value)

                if i not in whitep_counter_dict.keys():
                    prob_whitep_counter_list.append(no_information_value)
                else:
                    informative_value = round(
                        (whitep_counter_dict[i] + k) / (len(training_data_wrt_y) + len(given_data_name_labels) * k), 2)
                    if need_log == True:
                        informative_value = round(math.log(informative_value), 2)
                    prob_whitep_counter_list.append(informative_value)

            prob_of_fi_given_y.append([prob_blackp_counter_list, prob_whitep_counter_list])
        return prob_of_fi_given_y

    def probability_of_y_given_fi_bp(self, prob_fi_given_y, true_label_prob, test_matrices, given_data_name_labels):
        # test_feature_set = Data.blackPixelFeatureSet(0 ,test_matrices)
        test_feature_set = test_matrices
        predicted_probs = []
        predicted_label = []
        for i, x in enumerate(test_feature_set):
            prob_y_given_x = []
            max_val = -99999
            max_pos = -99999
            for y, label_value in enumerate(given_data_name_labels):
                prob_x_given_y = prob_fi_given_y[y][0][x[0]] * prob_fi_given_y[y][1][
                    x[1]]  # different if log valu is taken
                pred_val = prob_x_given_y * true_label_prob[y]
                prob_y_given_x.append(round(prob_x_given_y * true_label_prob[y], 9))
                if (pred_val > max_val):
                    max_val, max_pos = pred_val, y
            predicted_probs.append(prob_y_given_x)
            predicted_label.append(max_pos)
        return [predicted_probs, predicted_label]

    """Below code is for identifying better value of K """

    # needs to be tested   # this can be generalized
    def find_better_value_k(self, training_bwp_data_df, given_data_name_labels, matrix_element_count, k_list,
                            true_label_prob,
                            test_matrices, actual_pred_labels, need_log=False):
        accuaracy_list = []
        for k in k_list:
            prob_fi_given_y = self.probablity_of_fi_given_y(training_bwp_data_df, given_data_name_labels,
                                                            matrix_element_count, k, need_log=False)
            pred_labels = self.probability_of_y_given_fi(prob_fi_given_y, true_label_prob, test_matrices,
                                                         given_data_name_labels)
            accuaracy_list.append(self.evaluation_metric(actual_pred_labels, pred_labels))
        max_val = -1
        max_pos = -1
        for i, acc_val in enumerate(accuaracy_list):
            if (acc_val > max_val):
                max_val = acc_val
                max_pos = i
        return k_list[max_pos]

    def probability_of_fi_given_y_px(self, training_px_data, given_data_name_labels, k, need_log=True):
        prob_of_fi_given_y = []
        for var_y in given_data_name_labels:
            training_data_wrt_y = training_px_data.iloc[:, :-1][training_px_data.iloc[:, -1].values == var_y].values
            sum_of_black_per_pixels = np.sum(training_data_wrt_y, axis=0)
            sum_of_white_per_pixels = np.ones(len(sum_of_black_per_pixels)) * len(
                training_data_wrt_y) - sum_of_black_per_pixels
            prob_of_black_per_pixels = [round((X + k) / (len(training_data_wrt_y) + len(given_data_name_labels) * k), 4)
                                        for X in sum_of_black_per_pixels]
            prob_of_white_per_pixels = [round((X + k) / (len(training_data_wrt_y) + len(given_data_name_labels) * k), 4)
                                        for X in sum_of_white_per_pixels]
            if (need_log):
                prob_of_black_per_pixels = [round(math.log(X), 4) for X in prob_of_black_per_pixels]
                prob_of_white_per_pixels = [round(math.log(X), 4) for X in prob_of_white_per_pixels]
            prob_of_fi_given_y.append([prob_of_black_per_pixels, prob_of_white_per_pixels])
        return prob_of_fi_given_y

    def probability_of_y_given_fi_px(self, prob_fi_given_y, true_label_prob, test_matrices, given_data_name_labels):
        # test_feature_set = pixelFeatureSet(test_matrices)
        test_feature_set = test_matrices
        test_feature_set = pd.DataFrame(test_feature_set)
        predicted_probs = []
        predicted_label = []
        for i in range(len(test_feature_set.iloc[:, 0])):
            test_point = pd.DataFrame(test_feature_set.iloc[i, :])
            prob_y_given_x = []
            max_val = -99999
            max_pos = -99999
            for y, label_name in enumerate(given_data_name_labels):
                # we assume that we take log value always
                black_px_indexes = list(test_point[test_point.iloc[:, 0] == 1].index)
                white_px_indexes = list(test_point[test_point.iloc[:, 0] == 0].index)
                pred_value = round(math.log(true_label_prob[y]), 2) + np.sum(
                    np.array(prob_fi_given_y)[y][0][black_px_indexes]) + np.sum(
                    np.array(prob_fi_given_y)[y][1][white_px_indexes])
                prob_y_given_x.append(pred_value)
                if (pred_value > max_val):
                    max_val, max_pos = pred_value, label_name
            predicted_probs.append(prob_y_given_x)
            predicted_label.append(max_pos)
        return [predicted_probs, predicted_label]

    def process_execution(self, train_model, test_data, process_name, training_ratio, k):

        if (process_name == 'reduced_pixel'):
            start = timer()
            train_model.train_naive_model('pixel', k, training_ratio)
            end = timer()
            transformed_testdata = train_model.get_feature_selection('reduced_pixel',
                                                                     train_model.get_arg_s_for_feature_selection(
                                                                         'reduced_pixel', test_data.matrices,
                                                                         process='test'), process='test')
            predicted_information = train_model.probability_of_y_given_fi_px(train_model.train_information_matrix,
                                                                             train_model.true_label_prob,
                                                                             transformed_testdata,
                                                                             train_model.given_data_name_labels)
            return [train_model.evaluation_metric(test_data.labels, predicted_information[1]),end-start]

        elif (process_name == 'black_white_digit'):
            return "not Implemented yet"


if __name__ == '__main__':

    # digit_train_model = NaiveBayes(DataPath.getPath(0 ,'TRAINING_DIGIT_DATA_PATH'),DataPath.getPath(0,'TRAINING_DIGIT_LABEL_PATH'),
    #                                Training_Sample_Size, 28, 28, 'digit', 'black_count')
    #
    # digit_train_model.train_naive_model('black_count' , 2)
    #
    # digit_test_data = Data(DataPath.getPath(0,'TEST_DIGIT_DATA_PATH'), DataPath.getPath(0,'TEST_DIGIT_LABEL_PATH'),
    #                        Test_Sample_Size, 28, 28)
    #
    # predicted_information = digit_train_model.probability_of_y_given_fi_bp(digit_train_model.train_information_matrix ,
    #                                                digit_train_model.true_label_prob ,
    #                                                digit_test_data.matrices, digit_train_model.given_data_name_labels)
    #
    # print(digit_train_model.evaluation_metric( digit_test_data.labels , predicted_information[1]))
    #

    # digit_train_model = NaiveBayes(DataPath.getPath(0 ,'TRAINING_DIGIT_DATA_PATH'),DataPath.getPath(0,'TRAINING_DIGIT_LABEL_PATH'),Training_Sample_Size, 28, 28, 'digit', 'reduced_black_count')
    # digit_train_model.train_naive_model('black_count' , 2)
    # digit_test_data = Data(DataPath.getPath(0,'TEST_DIGIT_DATA_PATH'), DataPath.getPath(0,'TEST_DIGIT_LABEL_PATH'),Test_Sample_Size, 28, 28)
    # transformed_testdata = digit_train_model.get_feature_selection('reduced_black_count',digit_train_model.get_arg_s_for_feature_selection('reduced_black_count' ,digit_test_data.matrices, process = 'test'),process = 'test')
    # predicted_information = digit_train_model.probability_of_y_given_fi_bp(digit_train_model.train_information_matrix ,digit_train_model.true_label_prob ,transformed_testdata, digit_train_model.given_data_name_labels)
    # print(digit_train_model.evaluation_metric( digit_test_data.labels , predicted_information[1]))

    # digit dectection

    processes = ['reduced_pixel', 'black_white']
    datasets = ['digit', 'face']
    process_name = processes[0]
    data = datasets[1]
    num_of_iterations = 5
    training_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    final_prediction_stats = []
    K = 2

    if (data == 'digit'):
        Training_Sample_Size = 5000  #
        Test_Sample_Size = 1000  # 1000
        Validaion_Sample_Size = 1000
        # accuracies for k values in range(1,11) -> [0.747, 0.742, 0.74, 0.733, 0.731, 0.73, 0.73, 0.724, 0.719, 0.715]
        # validation set -> [0.792, 0.788, 0.786, 0.78, 0.778, 0.777, 0.775, 0.771, 0.764, 0.76]
        # best k value -> 1
        K = 1
        training_information_model = NaiveBayes(DataPath.getPath(0, 'TRAINING_DIGIT_DATA_PATH'),
                                                DataPath.getPath(0, 'TRAINING_DIGIT_LABEL_PATH'), Training_Sample_Size,
                                                28, 28,
                                                'digit', 'reduced_pixel')
        digit_test_data = Data(DataPath.getPath(0, 'TEST_DIGIT_DATA_PATH'),
                               DataPath.getPath(0, 'TEST_DIGIT_LABEL_PATH'),
                               Test_Sample_Size, 28, 28)

        digit_validate_data = Data(DataPath.getPath(0, 'VALIDATION_DIGIT_DATA_PATH'),
                                   DataPath.getPath(0, 'VALIDATION_DIGIT_LABEL_PATH'),
                                   Validaion_Sample_Size, 28, 28)

    elif (data == 'face'):
        Training_Sample_Size = 451  #
        Test_Sample_Size = 150  # 1000
        Validation_Sample_Size = 301
        # accuracies for k values in range(1,11) -> [0.894, 0.9, 0.9, 0.907, 0.907, 0.92, 0.92, 0.927, 0.92, 0.927]
        # validation set ->[0.887, 0.887, 0.887, 0.8904, 0.8904, 0.8904, 0.887, 0.8904, 0.884, 0.8804]
        # best k value is 8
        K = 8
        training_information_model = NaiveBayes(DataPath.getPath(0, 'TRAINING_FACE_DATA_PATH'),
                                                DataPath.getPath(0, 'TRAINING_FACE_LABEL_PATH'), Training_Sample_Size,
                                                60, 70,
                                                'face', 'reduced_pixel')
        digit_test_data = Data(DataPath.getPath(0, 'TEST_FACE_DATA_PATH'), DataPath.getPath(0, 'TEST_FACE_LABEL_PATH'),
                               Test_Sample_Size, 60, 70)

        digit_validate_data = Data(DataPath.getPath(0, 'VALIDATION_FACE_DATA_PATH'),
                                   DataPath.getPath(0, 'VALIDATION_FACE_LABEL_PATH'),
                                   Validation_Sample_Size, 60, 70)

    """Finding better value for K"""
    # counter = 0
    # acc_val = []
    # for k in range(1, 11):
    #   acc = training_information_model.process_execution(training_information_model, digit_validate_data, process_name,
    #                                                        1.0, k)
    #     acc_val.append(acc)
    #     counter += 1
    #     print(counter, end="")
    # print(acc_val)

    counter = 0
    for ratio in training_ratio_list:
        acc_val = []
        time_taken = []
        for i in range(num_of_iterations):
            digit_train_model = copy.deepcopy(training_information_model)
            acc = digit_train_model.process_execution(digit_train_model , digit_test_data , process_name , ratio , K)
            time_taken.append(acc[1])
            acc = acc[0]
            acc_val.append(acc)

        counter +=1
        print(counter, end='')
        final_prediction_stats.append([acc_val,statistics.mean(acc_val),statistics.stdev(acc_val),statistics.mean(time_taken)])

    f = open("naive_bayes_results.txt" , "a")
    f.writelines(pd.DataFrame(final_prediction_stats).to_string())
    plt.plot(pd.DataFrame(final_prediction_stats).iloc[:,1],training_ratio_list )

#    make classification reports based on the model