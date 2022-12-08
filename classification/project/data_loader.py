"""
A Data Loader module with utility functions  to read the images and convert them into matrix representing the pixels values of 0s and 1s

"""
import os
import sys
import zipfile
import numpy as np

class Data:

    def __init__(self, train_data_file, train_label_file, train_size, width, height):

        self.height = height
        self.width = width
        self.size = train_size
        self.matrices = self.process_raw_data_to_matrices(self.loadDataFile(train_data_file))
        self.labels = self.loadLabelsFile(train_label_file, train_size)

    def process_raw_data_to_matrices(self, raw_data):
        matrices = []
        for image in raw_data:
            matrix = [[0] * self.width for i in range(self.height)]

            for i, row in enumerate(image):
                for j, cell in enumerate(row):
                    if cell != " ":
                        matrix[i][j] = 1

            # for i in range(len(matrix)):
            #     for j in range(len(matrix[0])):
            #         if matrix[i][j] == 1:
            #             print("#", end="")
            #         else:
            #             print(" ", end="")
            #     print("\n")
            matrices.append(matrix)
        return matrices

    def readlines(self, filename):
        "Opens a file or reads it from the zip archive data.zip"
        if (os.path.exists(filename)):
            return [l[:-1] for l in open(filename).readlines()]
        else:
            z = zipfile.ZipFile('data.zip')
            return z.read(filename).split('\n')

    def loadDataFile(self, filename):
        """
        Reads n data images from a file and returns a list of Datum objects.

        (Return less then n items if the end of file is encountered).
        """

        fin = self.readlines(filename)
        fin.reverse()
        items = []
        for i in range(self.size):
            data = []
            for j in range(self.height):
                data.append(list(fin.pop()))
            if len(data[0]) < self.width - 1:
                # we encountered end of file...
                print("Truncating at %d examples (maximum)" % i)
                break
            # items.append(Datum(data, DATUM_WIDTH, DATUM_HEIGHT))
            items.append(data)
        return items

    def loadLabelsFile(self, filename, n):
        """
        Reads n labels from a file and returns a list of integers.
        """
        fin = self.readlines(filename)
        labels = []
        for line in fin[:min(n, len(fin))]:
            if line == '':
                break
            labels.append(int(line))
        return labels

    def pixelFeatureSet(self , matrices , actual_labels = None , attach_y = False):
        feature_data_matrix = []
        for i, matrix in enumerate(matrices):
            feature_set = []
            for row in matrix:
                feature_set.extend(row)
            if (attach_y == True):
                feature_set.append(actual_labels[i])
            feature_data_matrix.append(feature_set)
        return feature_data_matrix

    def blackPixelFeatureSet(self , matrices , actual_labels = None , attach_y = False ):
        feature_data_matrix = []
        num_of_pixels = len(matrices[0]) * len(matrices[0][0])
        for i, matrix in enumerate(matrices):
            pixel_count_list = []
            blackp_count = np.concatenate(matrix).sum()
            pixel_count_list.append(blackp_count)
            pixel_count_list.append(num_of_pixels - blackp_count)
            if (attach_y == True):
                pixel_count_list.append(actual_labels[i])
            feature_data_matrix.append(pixel_count_list)
        return feature_data_matrix
    def reducedGridFeatureSet(self , matrices , g_length , g_width , jump_value , black_white = False , reduced_noise = [3,False]):
        feature_data_matrix = []
        for matrix in matrices:
            reduced_feature_set = []
            for i in range(0, len(matrix) - g_width + 1, jump_value):
                row_feature_set = []
                for j in range(0, len(matrix[0]) - g_length + 1, jump_value):
                    sum_v = 0
                    for k in range(0, g_width):
                        for l in range(0, g_length):
                            sum_v += matrix[i + k][j + l]
                    if not black_white:
                        row_feature_set.append(sum_v)
                    else:
                        if not reduced_noise[1]:
                            if sum_v > 0: row_feature_set.append(1)
                            else : row_feature_set.append(0)
                        else:
                            if sum_v > reduced_noise[0]:  row_feature_set.append(1)
                            else: row_feature_set.append(1)
                reduced_feature_set.append(row_feature_set)
            feature_data_matrix.append(reduced_feature_set)
        return feature_data_matrix

    def trueLabelProbability(self ,actual_labels , given_data_name_labels):
        label_dict = {key: 0 for key in given_data_name_labels}
        for x in actual_labels:
            label_dict[x] += 1
        true_label_prob = np.array(list(label_dict.values())) / len(actual_labels)
        return true_label_prob


if __name__ == "__main__":
    Data("C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/trainingimages",
         "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/traininglabels", 25, 28, 28)

    #Data("/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrain",
    #    "/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrainlabels", 5, 60, 70)
