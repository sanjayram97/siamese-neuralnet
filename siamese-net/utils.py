from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
import os
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Lambda
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import config

def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def build_siamese_model(inputShape, embeddingDim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    # build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    return model

def fetch_images(img_folder, target_size = (28,28)):
  images = []
  labels = []
  classes = next(os.walk(img_folder))[1]
  class_ct = 0
  for i in classes:
    target_class_path = os.path.join(img_folder, i)
    for name in glob.glob(target_class_path+'/*'):
      img = cv2.imread(name)
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      resized_gray_img = cv2.resize(gray_img, target_size)
      images.append(resized_gray_img)
      labels.append(class_ct)
    class_ct+=1
  return np.array(images), np.array(labels)

def make_pairs(images, labels):
    # Image list to hold image, image pair
    # Label list to indicate if the image pair is positive/negative
    pairImages = []
    pairLabels = []
    
    # Unique classes
    numClasses = len(np.unique(labels))
    
    # Unique classes with the images indexes for each class
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    
    # loop over all images
    for idxA in range(len(images)):
        # get current image and its label
        currentImage = images[idxA]
        label = labels[idxA]
        
        # randomly pick an image of same label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        
        # Positive pair
        # Make image pair and label
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        
        
        # randomly pick image of different label
        negIdx = np.where(labels!=label)[0]
        negImage = images[np.random.choice(negIdx)]
        
        # Negative pair
        # Make image pair and label
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
        
    return np.array(pairImages), np.array(pairLabels)


def get_prediction_scores(model, anchor, images, labels, target_size=(28,28)):
  # preprocess anchor image
  h,w = target_size
  _,_,c = anchor.shape
  if c != 1:
    anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
  anchor = cv2.resize(anchor, (h,w))
  print('---')
  print(anchor.shape)
  print('---')
  UniqueClasses = np.unique(labels)
  numClasses = len(UniqueClasses)
  idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
  similarity_score = {}
  scores = []
  anchor = anchor.reshape(1, h, w, 1)
  for idx,i in enumerate(idx):
    img = np.random.choice(i)
    print(img)
    tr_img = images[img].reshape(1, h, w, 1)
    score = model.predict([anchor, tr_img])
    similarity_score[idx] = score
    scores.append(score)
  return similarity_score, np.argmax(scores)