import cv2
import mediapipe as mp
import time
import os

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

img_no = 1
output_directory = "dataset/negative"

# Ensure the directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

while True:
    success, img = cap.read()
    if not success:
        continue
    
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpHands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        print("Hands detected")
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [lm for lm in hand_landmarks.landmark]

        x_min = int(min(landmark.x * w for landmark in landmarks))
        x_max = int(max(landmark.x * w for landmark in landmarks))
        y_min = int(min(landmark.y * h for landmark in landmarks))
        y_max = int(max(landmark.y * h for landmark in landmarks))

        # img = cv2.flip(img, 1)
        cropped_img = img[y_min - 20: y_max + 20, x_min - 20: x_max + 20]

        cv2.imshow("img", cropped_img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f"Image_{img_no}.jpg"
            filepath = os.path.join(output_directory, filename)
            
            # Try to save the image
            success = cv2.imwrite(filepath, cropped_img)
            if success:
                print(f"Image {filename} saved successfully.")
            else:
                print(f"Error saving image {filename}.")
            
            img_no += 1
    else:
        print("Hands not detected")

cap.release()
cv2.destroyAllWindows()
