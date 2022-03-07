import pickle
import cv2 as cv
import numpy as np
from CharacterIdentification import feature_type
def getFeatures():
    feature = feature_type.what_feature()
    if feature == "projection":
        file = open("projection_features.pkl",'rb')
    if feature == "knn":
        file = open("knn_features.pkl",'rb')
    if feature == "pca":
        file = open("pca_vectors.pkl",'rb')
    obj = pickle.load(file)
    return obj
    pass