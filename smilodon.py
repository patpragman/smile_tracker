import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# Blue color in BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

with open("linear_regression.pkl", "rb") as file:
    classifier = pickle.load(file)
    print('linear  regression model loaded...')

def detect_smile(gray, x, y, w, h) -> bool:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1, 1).T
    return classifier.predict(face)

emotional_state = 0
last_detected_face = (None, None, None, None)
def detect_face(gray, frame):
    """
    this nice code comes from geeks4geeks stackoverflow and a mix of crazy tinkering
    """

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # detect smiles

        # draw a bounding box around the face
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), BLUE, 2)

        if detect_smile(gray, x, y, w, h):
            cv2.putText(frame, ':)', (x - 20, y), font, 1, GREEN, 2, cv2.LINE_AA)
            cv2.imwrite("last_smile.png", frame)
        else:
            cv2.putText(frame, ':|', (x - 20, y), font, 1, BLUE, 2, cv2.LINE_AA)
            cv2.imwrite('last_meh.png', frame)





    return frame

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:
    # reads frames from a camera
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = detect_face(gray, frame)

    # Display an image in a window
    cv2.imshow('img', frame)



    # Wait for Esc key to stop
    k = cv2.waitKey(1) # current catched key
    if k == 27:  # escape key.
        break
# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
