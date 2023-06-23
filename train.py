import pandas as pd
import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from model import SmallerVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import backend as K

def train():
    EPOCHS = 20
    INIT_LR = 1e-3
    BS = 32
    IMAGE_DIMS = (640, 640, 3)

    jpeg_list = list(pd.read_csv('df.csv')['images'])
    imagePaths = ['/Users/luca/Documents/Desktop/approved/' + s + '_1.jpg' for s in jpeg_list]

    with open('labels.csv', newline='') as f:
        reader = csv.reader(f)
        labels = list(reader)

    random.seed(42)
    c = list(zip(imagePaths, labels))
    random.shuffle(c)
    imagePaths, labels = zip(*c)

    data = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(mlb.classes_))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    history = model.fit(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)
    model.save("multilabel_keras.h5")
