import cv2 as cv
import numpy as np
from CharacterIdentification import LabelReader
from CharacterIdentification import ImageReader
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors=6)