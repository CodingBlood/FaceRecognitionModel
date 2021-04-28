import cv2
import os
def CapNCrop(loc):
    from PIL import Image
    from numpy import asarray
    # load the image
    img = cv2.imread(loc)
    # convert image to numpy array
    # img = asarray(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = img[y:y+270, x:x+270]
        fname = './Datasets/Test/pred/1.jpg'
        cv2.imwrite(filename=fname, img=roi)

CapNCrop('./Datasets/Test/Kartik/2021-04-28-165253.jpg')