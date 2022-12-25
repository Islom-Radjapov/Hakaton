import cv2
import numpy as np
from keras import models
from detect_pose import mp_holistic, mediapipe_detection, draw_styled_landmarks


cap = cv2.VideoCapture(0)
model = models.load_model('read_mation.model')
# img_dims = (224, 224, 3)

labels = ["Assalomu alekum", "men", 'ismim', 'Ali', 'dasturchi', 'yoshim', '17', 'A', 'L', 'I']

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

            conf = model.predict(crop1)[0]
            idx = np.argmax(conf)
            label = labels[idx]
            print(label)

            # cv2.rectangle(imgOutput, (300, 400),
            #               (350, 475), (0, 0, 0), cv2.FILLED)

            cv2.putText(imgOutput, labels[idx], (305, 460), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 4)

        cv2.imshow("Detect", crop)


        cv2.imshow("Image", imgOutput)
        if key == ord('q'):
            break