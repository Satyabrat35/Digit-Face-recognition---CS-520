import os
import zipfile


class Data:

    def __init__(self, train_data_file, train_label_file, train_size, width, height):

        self.height = height
        self.width = width
        self.size = train_size
        self.matrices = self.process_raw_data_to_matrices(self.loadDataFile(train_data_file))
        self.labels = self.loadLabelsFile(train_label_file, train_size)
        print('hello')

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


if __name__ == "__main__":
    # Data("/Users/pranoysarath/Downloads/classification/data/digitdata/trainingimages",
    #      "/Users/pranoysarath/Downloads/classification/data/digitdata/traininglabels", 5, 28, 28)

    Data("/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrain",
         "/Users/pranoysarath/Downloads/classification/data/facedata/facedatatrainlabels", 5, 60, 70)
