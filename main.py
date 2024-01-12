'''
Hand Tracking = Palm Detection (cropped img of hand) + Hand Landmarks (20 different landmarks)
'''

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("index_finger_cnn.keras")

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

trail = []

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpHands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_height, img_width, _ = img.shape
    if results.multi_hand_landmarks:
        print("Hands detected")
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [lm for lm in hand_landmarks.landmark]
            indexf_x, indexf_y, _ = landmarks[8].x, landmarks[8].y, landmarks[8].z
            
            x, y = int(indexf_x * img_width), int(indexf_y * img_height)

        x_min = int(min(landmark.x * img_width for landmark in landmarks))
        x_max = int(max(landmark.x * img_width for landmark in landmarks))
        y_min = int(min(landmark.y * img_height for landmark in landmarks))
        y_max = int(max(landmark.y * img_height for landmark in landmarks))

        cropped_img = img[y_min - 20: y_max + 20, x_min - 20: x_max + 20]
        x_test = np.array([cv2.resize(cropped_img, (128, 256))])

        y_pred = model.predict(x_test)[0][0]
        if y_pred > 0.6:
            trail.append((x, y))
    else:
        print("Hands not detected")

    for point in trail:
        cv2.circle(img, (point[0], point[1]), 10, (0, 0, 0), -1)

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


'''
For bounding box:
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for rect in results.hand_rects:
  print(rect)
'''