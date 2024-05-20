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



