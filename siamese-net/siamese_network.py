from utils import *
# from siamese-net.config import *
IMG_SHAPE = (28, 28, 1)
TRAIN_IMG_PATH = r"C:\Users\ASUS\Documents\Projects\Siamese_network\package\output"
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Lambda
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

trainX, trainY = fetch_images(TRAIN_IMG_PATH, target_size = (28,28))
testX, testY = fetch_images(config.TRAIN_IMG_PATH, target_size = (28,28))

pairTrain, labelTrain = make_pairs(trainX, trainY)
pairTest, labelTest = make_pairs(testX, testY)

images = []

trainX = trainX / 255.0
testX = testX / 255.0

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)

# configure the siamese network
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# construct the siamese network

distance = Lambda(euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# training
history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
                    validation_data = ([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
                    epochs = 100)


anchor = cv2.imread(r'C:\Users\ASUS\Documents\Projects\Siamese_network\package\siamese-image-valid\saran\2.jpeg')
similarity_dict, target = model_instance.make_prediction(anchor)
print(similarity_dict)