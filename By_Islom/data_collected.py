from detect_hand import find_hand
import cv2
import uuid
from time import sleep

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
out = cv2.VideoWriter(fr'C:\Users\islom\PycharmProjects\Hakaton\Data\dasturlash\{uuid.uuid1()}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
counter = 0
# sleep(5)
while counter != 30:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    detect_hand = find_hand(image=image)
    cv2.imshow("Image", detect_hand)
    if counter == 0:
        sleep(2)
    out.write(detect_hand)
    counter += 1
    print(counter)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()