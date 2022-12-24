import os
import cv2

folder = os.listdir(r'C:\Users\islom\PycharmProjects\Hakaton\Data\avi')
list_videos = fr'C:\Users\islom\PycharmProjects\Hakaton\Data\avi\{folder[1]}'
cap = cv2.VideoCapture(list_videos)
counter = 0
while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    print(counter)
    counter += 1
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()