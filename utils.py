import csv
import cv2
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array

def load_labels_csv(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        labels = list(reader)
    return labels

def preprocess_image(image_path, image_dims):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def decode_predictions(predictions, mlb):
    labels = mlb.inverse_transform(predictions)
    return labels
