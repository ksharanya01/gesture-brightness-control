import math
import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

MIN_DIST = 18     # your value: fingers touching
MAX_DIST = 215    # your value: fingers wide apart

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

            height, width, _ = frame.shape
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            x1, y1 = int(thumb.x * width), int(thumb.y * height)
            x2, y2 = int(index.x * width), int(index.y * height)

            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            distance = math.hypot(x2 - x1, y2 - y1)

            # map distance (18-215) to brightness (0-100)
            level = int(np.interp(distance, [MIN_DIST, MAX_DIST], [0, 100]))

            try:
                sbc.set_brightness(level)
            except Exception as e:
                print("Brightness error:", e)

            cv2.putText(frame, f"Brightness: {level}%", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()