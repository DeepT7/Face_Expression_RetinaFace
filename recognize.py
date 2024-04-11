from tensorflow.keras.models import load_model 
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu',))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(7, activation='softmax'))

    return model 

def image_processing(image):
    # img = load_img(img_path, target_size=(48, 48))
    img = cv2.resize(image, (48, 48))
    img = img_to_array(img)
    if img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)

    img = img[..., 0][np.newaxis, ...]
    # img = tf.expand_dims(img, axis = 0)
    img = img.astype('float32') / 255.0
    return img 

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0 



