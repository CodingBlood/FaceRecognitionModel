import cv2
import os
from tensorflow import keras
import numpy as np
import tensorflow as tf
# import time
import cv2
# import FaceDetectionAndCroppingTest
from PIL import Image
from numpy import asarray
def CapNCrop():
    # a = cv2.imread('./Datasets/Train/Kartik/1.jpg')
    # print(a.shape)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    # cap = cv2.VideoCapture('filename.mp4')
    i = 1
    m = 0
    k = 0
    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (270,270))
            print(roi.shape)
            data = roi.reshape(-1, 270, 270, 3)
            # print(data.shape)
            model = keras.models.load_model('facefeatures_new_model_new.h5')
            predictions = model.predict(data)
            # print(predictions)
            classes = np.argmax(predictions)
            print(classes)
            if classes == 0:
                cv2.putText(img, 'KARTIK', (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                k += 1
            else:
                cv2.putText(img, 'MOHAN', (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                m+=1
            i += 1
            print(i)
            if i == 30:
                k = 27
                break

        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
    if k>m:
        # imge = cv2.VideoCapture(0)
        # img = cv2.cvtColor(imge.read(), cv2.COLOR_BGR2GRAY)
        cv2.putText(img, 'Hey Kartik', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('imges', img)
        cv2.waitKey()
    else:
        # imge = cv2.VideoCapture(0)
        # img = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
        cv2.putText(img, 'Hey Mohan', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('imges', img)
        cv2.waitKey()

CapNCrop()










#
# # load the image
# # FaceDetectionAndCroppingTest.CapNCrop('./Datasets/Test/Kartik/2021-04-28-165253.jpg')
# data = cv2.imread('./Datasets/Test/pred/1.jpg')
# cv2.imshow("img to be predicted",data)
# cv2.waitKey()
# data=data.reshape(-1,270,270,3)
# print(data.shape)
# model = keras.models.load_model('facefeatures_new_model.h5')
# predictions = model.predict(data)
# print(predictions)
# classes = np.argmax(predictions)
# if classes == 1:
#     print("Mohan")
# else:
#     print("Kartik")
# data = cv2.imread('./Datasets/Test/pred/2.jpg')
# cv2.imshow("img to be predicted",data)
# cv2.waitKey()
# data=data.reshape(-1,270,270,3)
# print(data.shape)
# model = keras.models.load_model('facefeatures_new_model.h5')
# predictions = model.predict(data)
# print(predictions)
# classes = np.argmax(predictions)
# if classes == 1:
#     print("Mohan")
# else:
#     print("Kartik")
