import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from keras import models

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = models.load_model('read_latter.model')

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                # preprocessing for hand detection model
                find_hand = cv2.resize(imgWhite, (96, 96))
                find_hand = find_hand.astype("float") / 255
                # face_crop = img_to_array(face_crop)   # ???
                find_hand = np.expand_dims(find_hand, axis=0)
                conf = model.predict(find_hand)[0]
                idx = np.argmax(conf)
                label = labels[idx]

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                # preprocessing for hand detection model
                find_hand = cv2.resize(imgWhite, (96, 96))
                find_hand = find_hand.astype("float") / 255
                # face_crop = img_to_array(face_crop)   # ???
                find_hand = np.expand_dims(find_hand, axis=0)
                conf = model.predict(find_hand)[0]
                idx = np.argmax(conf)
                label = labels[idx]

            cv2.imshow("ImageWhite", imgWhite)

            cv2.rectangle(imgOutput, (300, 400),
                          (350, 475), (0, 0, 0), cv2.FILLED)

            cv2.putText(imgOutput, labels[idx], (305, 460), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 4)

        except:
            pass

    key = cv2.waitKey(1)
    cv2.imshow("Image", imgOutput)
    if key == ord('q'):
        break