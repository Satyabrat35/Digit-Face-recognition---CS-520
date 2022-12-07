class DataPath():

    ### Just in case if the init instantiaition fails
    def getPath(self ,data_variable_path):
        if data_variable_path == "TRAINING_DIGIT_DATA_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/trainingimages"
        if data_variable_path == "TRAINING_DIGIT_LABEL_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/traininglabels"
        if data_variable_path == "TEST_DIGIT_DATA_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/testimages"
        if data_variable_path == "TEST_DIGIT_LABEL_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/testlabels"
        if data_variable_path == "VALIDATION_DIGIT_DATA_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/validationimages"
        if data_variable_path == "VALIDATION_DIGIT_LABEL_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/digitdata/validationlabels"

        if data_variable_path == "TRAINING_FACE_DATA_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/facedata/facedatatrain"
        if data_variable_path == "TRAINING_FACE_LABEL_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/facedata/facedatatrainlabels"
        if data_variable_path == "TEST_FACE_DATA_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/facedata/facedatatest"
        if data_variable_path == "TEST_FACE_LABEL_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/facedata/facedatatestlabels"
        if data_variable_path == "VALIDATION_FACE_DATA_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/facedata/facedatavalidation"
        if data_variable_path == "VALIDATION_FACE_LABEL_PATH":
            return "C:/Users/Royale121/PycharmProjects/Digit-Face-recognition---CS-520/classification/data/facedata/facedatavalidationlabels"

