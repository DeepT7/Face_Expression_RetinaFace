import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.python.ops.numpy_ops import np_config

classes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']

IMAGE_SIZE = 112
def image_processing(image):
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    cv2.imwrite('cropped_facec.png', img)

    img = img_to_array(img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis = 0)
    return img

