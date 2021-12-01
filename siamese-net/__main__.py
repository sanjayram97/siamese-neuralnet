from utils import fetch_images, make_pairs, build_siamese_model, euclidean_distance, get_prediction_scores
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Lambda
import tensorflow.keras.backend as K
import cv2

class SiameseNet:
    def __init__(self, trainPath, testPath, EPOCH = 10, image_shape=(28,28), loss = "binary_crossentropy", optimizer = "adam", metrics=["accuracy"]):
        
        self.trainPath = trainPath
        self.testPath = testPath
        self.EPOCH = EPOCH
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.image_shape = image_shape
        print('Model Instantiated')
        
    def prepare_dataset(self):
        
        trainX, trainY = fetch_images(self.trainPath, target_size = self.image_shape)
        testX, testY = fetch_images(self.testPath, target_size = self.image_shape)
        print('Initial dataset prepared')
        trainX = trainX / 255.0
        testX = testX / 255.0
        
        trainX = np.expand_dims(trainX, axis=-1)
        testX = np.expand_dims(testX, axis=-1)
        
        (self.pairTrain, self.labelTrain) = make_pairs(trainX, trainY)
        (self.pairTest, self.labelTest) = make_pairs(testX, testY)
        
        self.trainX = trainX
        self.trainY = trainY
        
        return (self.pairTrain, self.labelTrain), (self.pairTest, self.labelTest)
    
    def config_network(self):
        print('Siamese network implementation')
        third_channel = (1,)
        
        imgA = Input(shape = self.image_shape + third_channel)
        imgB = Input(shape = self.image_shape + third_channel)
        featureExtractor = build_siamese_model(self.image_shape + third_channel)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)

        # construct the siamese network
        distance = Lambda(euclidean_distance)([featsA, featsB])
        outputs = Dense(1, activation="sigmoid")(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        self.model = model
        print('Siamese network implemented')
        return model
    
    def train_model(self, model):
        self.model = model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # training
        history = self.model.fit([self.pairTrain[:, 0], self.pairTrain[:, 1]], self.labelTrain[:],
                            validation_data = ([self.pairTest[:, 0], self.pairTest[:, 1]], self.labelTest[:]),
                            epochs = self.EPOCH)
        
        return history
    
    def make_prediction(self, anchor):
        return get_prediction_scores(self.model, anchor, self.trainX, self.trainY)
    

if __name__ == "__main__":
    model_instance = SiameseNet(r'C:\Users\ASUS\Documents\Projects\Siamese_network\package\siamese-image-input', 
                                r'C:\Users\ASUS\Documents\Projects\Siamese_network\package\siamese-image-valid')
    (pairTrain, labelTrain), (pairTest, labelTest) = model_instance.prepare_dataset()
    model = model_instance.config_network()
    history = model_instance.train_model(model)
    
    anchor = cv2.imread(r'C:\Users\ASUS\Documents\Projects\Siamese_network\package\siamese-image-valid\saran\2.jpeg')
    similarity_dict, target = model_instance.make_prediction(anchor)
    print(similarity_dict)