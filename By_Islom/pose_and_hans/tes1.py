import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np


from detect_pose import mp_holistic, mediapipe_detection, draw_styled_landmarks
labels = ["Assalomu alekum", "men", 'ismim', 'Ali', 'dasturchi', 'yoshim', '17', 'A', 'L', 'I']

cap = cv2.VideoCapture(0)
classifier = Classifier("read_mation.h5", "labels.txt")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        status, image = cap.read()
        imgOutput = image.copy()

        # Make detections
        image, results = mediapipe_detection(image, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)
        key = cv2.waitKey(1)

        crop = cv2.resize(image, (250, 200))

        if results.left_hand_landmarks or results.right_hand_landmarks:
            crop1 = crop.astype("float") / 255
            crop1 = np.expand_dims(crop1, axis=0)

            prediction, index = classifier.getPrediction(imgOutput, draw=False)
            print(prediction, index)

            # cv2.rectangle(imgOutput, (300, 400),
            #               (350, 475), (0, 0, 0), cv2.FILLED)

            cv2.putText(imgOutput, labels[index], (305, 460), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 4)

        cv2.imshow("Detect", crop)

        cv2.imshow("Image", imgOutput)
        if key == ord('q'):
            break