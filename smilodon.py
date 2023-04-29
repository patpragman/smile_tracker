import cv2
import pickle

import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# Blue color in BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

with open("logistic_regression.pkl", "rb") as file:
    binary_classifier = pickle.load(file)
    print('Logistic regression model (binary) loaded...')

with open("logistic_regression_2.pkl", "rb") as file:
    multi_class_classifier = pickle.load(file)
    print('Logistic regression model (multi-class) loaded...')


def detect_smile(gray, x, y, w, h) -> bool:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 1).T
    return binary_classifier.predict(face)

def detect_all(gray, x, y, w, h) -> int:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 1).T
    return multi_class_classifier.predict(face)

def binary_labeler(gray_scale_image, frame_to_edit, x, y, w, h) -> np.ndarray:
    """detect only smile or no smile"""

    if detect_smile(gray_scale_image, x, y, w, h):
        cv2.putText(frame_to_edit, ':)', (x - 20, y), font, 1, GREEN, 2, cv2.LINE_AA)
        # draw a bounding box around the face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), GREEN, 2)
        cv2.imwrite("last_smile.png", frame_to_edit)
    else:
        cv2.putText(frame_to_edit, ':|', (x - 20, y), font, 1, BLUE, 2, cv2.LINE_AA)
        # draw a bounding box around the face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), BLUE, 2)
        cv2.imwrite('last_meh.png', frame_to_edit)
    return frame_to_edit


def multiclass_labeler(gray_scale_image, frame_to_edit, x, y, w, h) -> np.ndarray:
    """detect happy, sad, meh"""

    if detect_all(gray_scale_image, x, y, w, h) == 1:
        cv2.putText(frame_to_edit, ':)', (x - 20, y), font, 1, GREEN, 2, cv2.LINE_AA)
        # draw a bounding box around the face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), GREEN, 2)
        cv2.imwrite("last_smile.png", frame_to_edit)
    elif detect_all(gray_scale_image, x, y, w, h) == 0:
        cv2.putText(frame_to_edit, ':|', (x - 20, y), font, 1, BLUE, 2, cv2.LINE_AA)
        # draw a bounding box around the face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), BLUE, 2)
        cv2.imwrite('last_meh.png', frame_to_edit)
    else:
        cv2.putText(frame_to_edit, ':(', (x - 20, y), font, 1, RED, 2, cv2.LINE_AA)
        # draw a bounding box around the face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), RED, 2)
        cv2.imwrite('last_sad.png', frame_to_edit)

    # draw a bounding box around the face
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), BLUE, 2)

    return frame_to_edit


def detect_face(gray, frame, multi_class_mode=False):
    """
    this nice code comes from geeks4geeks stackoverflow and a mix of crazy tinkering
    """
    if not multi_class_mode:
        cv2.putText(frame, 'Binary Detector', (0, 20), font, 1, BLUE, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Multi-Class Detector', (0, 20), font, 1, BLUE, 2, cv2.LINE_AA)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # detect smiles

        if not multi_class_mode:
            binary_labeler(gray, frame, x, y, w, h)
        else:
            multiclass_labeler(gray, frame, x, y , w, h)

    cv2.putText(frame, 'Press Space to Flip Mode, ESC to quit.', (0, frame.shape[0] - 20), font, 1, BLUE, 2, cv2.LINE_AA)

    return frame

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
multi_class_mode = False
while 1:
    # reads frames from a camera
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = detect_face(gray, frame, multi_class_mode=multi_class_mode)

    # Display an image in a window
    cv2.imshow('img', frame)



    # Wait for Esc key to stop
    k = cv2.waitKey(1) # current catched key
    if k == 27:  # escape key.
        # Close the window
        break
    elif k == 32:
        multi_class_mode = not multi_class_mode

cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
