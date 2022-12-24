import os
import cv2
from time import sleep
folder = os.listdir(r'C:\Users\islom\PycharmProjects\Hakaton\Data\dasturlash')
print(folder[0])
list_videos = fr'C:\Users\islom\PycharmProjects\Hakaton\Data\dasturlash\{folder[0]}'
cap = cv2.VideoCapture(list_videos)
counter = 0
while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    print(counter)
    counter += 1
    sleep(0.08)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()