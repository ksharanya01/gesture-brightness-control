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
    frame = cv2.flip(frame, 1)                          # mirror the image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe needs RGB

    result = hands.process(frameRGB)                    # find the hand

    if result.multi_hand_landmarks:                     # true only if a hand was found
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()