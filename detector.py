import cv2
from keras.models import load_model
import numpy as np
import json


def main_app(name):
    labels = open('class_labels.json')
    labels = json.load(labels)
    id = labels[name]
    print(id)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = load_model('model.h5')
    cap = cv2.VideoCapture(0)
    pred = 0
    while True:
        ret, frame = cap.read()
        # default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale face to RGB
            face = np.expand_dims(face, axis=0)
            face = face / 255.0
            prediction = recognizer.predict(face)[0]
            confidence = np.max(prediction)
            predicted_id = np.argmax(prediction)
            confidence = 100 - int(confidence)
            if confidence > 80:
                # if u want to print confidence level
                # confidence = 100 - int(confidence)
                if predicted_id == id:
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = cv2.putText(frame, name, (x, y - 4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:
                    text = "UnknownFace"
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    frame = cv2.putText(frame, text, (x, y - 4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("image", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


