import mediapipe as mp
import cv2
import numpy as np
import time
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
score = 0

x_enemy = random.randint(50, 600)
y_enemy = random.randint(50, 400)

def enemy():
    global score, x_enemy, y_enemy
    cv2.circle(image, (x_enemy, y_enemy), 25, (0, 200, 0), 5)

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        imageHeight, imageWidth, _ = image.shape

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        color = (255, 0, 255)
        cv2.putText(image, "Score", (480, 30), font, 1, color, 4, cv2.LINE_AA)
        cv2.putText(image, str(score), (590, 30), font, 1, color, 4, cv2.LINE_AA)

        enemy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for point in mp_hands.HandLandmark:
                    normalized_landmark = hand_landmarks.landmark[point]
                    pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(
                        normalized_landmark.x, normalized_landmark.y, imageWidth, imageHeight)

                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        if pixel_coordinates_landmark:
                            cv2.circle(image, (pixel_coordinates_landmark[0], pixel_coordinates_landmark[1]), 25, (0, 200, 0), 5)
                            if abs(pixel_coordinates_landmark[0] - x_enemy) < 10:
                                print("found")
                                x_enemy = random.randint(50, 600)
                                y_enemy = random.randint(50, 400)
                                score += 1
                                cv2.putText(image, "Score", (100, 100), font, 1, color, 4, cv2.LINE_AA)
                                enemy()

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == 27:
            print(score)
            break

    video.release()
    cv2.destroyAllWindows()