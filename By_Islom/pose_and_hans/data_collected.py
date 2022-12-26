import cv2
import uuid
from detect_pose import mp_holistic, mp_drawing, mediapipe_detection, draw_landmarks, draw_styled_landmarks

# save videos
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
counter = 0

folder = r'C:\Users\islom\PycharmProjects\Hakaton\Data'

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        status, image = cap.read()
        key = cv2.waitKey(1)
        if not status:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)


        # Draw landmarks
        draw_styled_landmarks(image, results)

        cv2.imshow("Image", image)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(fr'{folder}\{uuid.uuid1()}.jpg', image)
            print(counter)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()