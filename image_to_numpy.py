import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

positive_path = os.path.join(os.getcwd(), "dataset/positive/")
negative_path = os.path.join(os.getcwd(), "dataset/negative/")


x_data = []
y_data = []

# positive
for img in os.listdir(positive_path):
    image_path = os.path.join(positive_path, img)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))

    x_data.append(img)
    y_data.append(1)

# negative
for img in os.listdir(negative_path):
    image_path = os.path.join(negative_path, img)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))

    x_data.append(img)
    y_data.append(0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)