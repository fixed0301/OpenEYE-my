import numpy as np
import os
import cv2
from keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
class_names = ['0 Opened', '1 Closed']
model_path = "model/final_model.h5"
def classify_img(one_eye_img): # input은 한 쪽 눈 이미지
    model = load_model(f"{model_path}", compile=False)
    img = cv2.resize(one_eye_img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)

    img = (img / 127.5) - 1

    prediction = model.predict(img)
    index = np.argmax(prediction)

    class_name = class_names[index]
    confidence_score = prediction[0][index]
    classified = class_name[2:]

    return classified
img = cv2.imread("E:\\2023\\2023_1_1\comp\data_processing\\facial_image\input_0\image_0\\face_8\l_eye.jpg")
print(classify_img(img))

