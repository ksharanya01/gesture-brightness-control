import math
import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=1
)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

            height, width, _ = frame.shape      # picture size in pixels

            thumb = hand.landmark[4]            # thumb tip
            index = hand.landmark[8]            # index fingertip

            # convert 0-1 fractions to real pixel positions
            x1, y1 = int(thumb.x * width), int(thumb.y * height)
            x2, y2 = int(index.x * width), int(index.y * height)

            # green circles on both tips, and a line between them
            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            distance = math.hypot(x2 - x1, y2 - y1)
            print(int(distance))

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()