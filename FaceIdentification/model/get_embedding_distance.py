# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet


def face_detection(imdir, img_size):
    # 读入一张图片
    img = cv2.imread(imdir)

    # 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征
    face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
    faces = face_engine.detectMultiScale(img, scaleFactor=1.3)

    # 对每一张脸，进行如下操作
    for (x, y, w, h) in faces:
        # 画出人脸框，蓝色（BGR色彩体系），画笔宽度为2
        cen_x = int(x + w / 2)
        cen_y = int(y + h / 2)
        x = int(cen_x - 0.65 * w)
        y = int(cen_y - 0.7 * h)
        w = int(2 * 0.65 * w)
        h = int(2 * 0.65 * h)
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = cv2.resize(img[y: y + h, x: x + w], (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        return face


image_size = 200
modeldir = "./model_check_point/20170512-110547.pb"
image_name1 = "Mrhuo.jpg"
image_name2 = "zhang_train.jpg"

print("loading facenet embedding...")
tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

print("facenet embedding complete...")

scaled_reshape = []

# image1
image1_p = face_detection(image_name1, image_size)
image1 = facenet.prewhiten(image1_p)
scaled_reshape.append(image1.reshape(-1, image_size, image_size, 3))
emb_array1 = np.zeros((1, embedding_size))
emb_array1[0, :] = \
    sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]

image2_p = face_detection(image_name2, image_size)
image2 = facenet.prewhiten(image2_p)
scaled_reshape.append(image2.reshape(-1, image_size, image_size, 3))
emb_array2 = np.zeros((1, embedding_size))
emb_array2[0, :] = \
    sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False})[0]

dist = np.sqrt(np.sum(np.square(emb_array1[0] - emb_array2[0])))
print("128D Euclidean Distance：%f " % dist)
cv2.imshow("image1", image1_p)
cv2.imshow("image2", image2_p)
cv2.waitKey(0)
