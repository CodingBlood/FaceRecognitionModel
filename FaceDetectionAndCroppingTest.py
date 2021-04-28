import cv2
import os
def CapNCrop(name):
    os.mkdir('./Datasets/Train/' + name)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    # cap = cv2.VideoCapture('filename.mp4')
    i=1
    k=1
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
            fname = './Datasets/Train/' + name + '/' + str(i) + ".jpg"
            print(fname)
            cv2.imwrite(filename=fname, img=roi)
            i += 1
            if i==151:
                k=27
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        cv2.waitKey(30)
        if k == 27:
            break
    # Release the VideoCapture object
    cap.release()
def main():
    name = str(input("Enter Your Name to Save Your Face Data And Make Sure No One Else Is In Frame!!"))
    try:
        CapNCrop(name)
    except:
        print("UserName Exists")
        main()
main()