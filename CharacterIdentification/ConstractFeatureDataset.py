from CharacterIdentification import ConfigReader as conf
from CharacterIdentification import words_location as feature_extract
import os
import cv2 as cv
import pickle
import numpy as np
from CharacterIdentification import feature_type
from sklearn.decomposition import PCA
what_feature = feature_type.what_feature()
given_char = set("学号：学院：光电工程学院姓名：江笑语发卡日期：-1234567890")
dir,_ = conf.get_dir_Chinese_Characters()
images_base = dir + "\\train"
image_files = os.listdir(images_base)
features = []
length = len(image_files)
count = 0
flatten_vectors = None
for image_file in image_files:
    image_dir = images_base + "\\" + image_file +"\\"+ "simkai.ttf.png"
    im = cv.imread(image_dir)
    im = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
    _, im = cv.threshold(im, 0,255,cv.THRESH_OTSU)

    # normalization
    # max_v,min_v = np.max(im), np.min(im)
    # N_im = (im - min_v) / (max_v - min_v)
    N_im = 255-im
    # cv.imshow("a",N_im)
    # cv.waitKey(500)
    if what_feature == "projection":
        feature = feature_extract.extract_pojections(N_im)
        features.append(feature)
    elif what_feature == "knn":
        feature = feature_extract.extract_knn(N_im)
        features.append(feature)
    elif what_feature == "pca":
        N_im = cv.resize(N_im, (20,20))
        flatten_vector = N_im.reshape(-1, 1)
        if flatten_vectors is None:
            flatten_vectors = flatten_vector
        else:
            flatten_vectors = np.hstack((flatten_vectors,flatten_vector))
    else:
        exit("unavailable features")

    count += 1
    if count %100 == 0 :
        print(str(count) +'/'+str(length))
if what_feature == "projection":
    pickle.dump(features, open('projection_features.pkl', 'wb'))
if what_feature == "knn":
    pickle.dump(features, open('knn_features.pkl', 'wb'))
if what_feature == "pca":
    top_K = 100
    cov_matirx = np.dot(flatten_vectors.T, flatten_vectors)
    vals, vecs = np.linalg.eig(cov_matirx)
    # idx = 0
    # top_idxes = []
    # top_val = []
    # for K in range(top_K):
    #     max_val_idx = 0
    #     for val_idx in range(len(vals)):
    #         if vals[max_val_idx] < vals[val_idx] and val_idx not in top_idxes:
    #             max_val_idx = val_idx
    #     top_idxes = [max_val_idx] + top_idxes
    # for val_idx in top_idxes:
    #     top_val.append(vals[val_idx])
    pickle.dump(vecs[:top_K], open('pca_vectors.pkl','wb'))
    pass
