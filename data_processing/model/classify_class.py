import numpy as np
import os
from keras.models import load_model
from PIL import Image, ImageOps
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

model = load_model("final_model.h5", compile=False)

image_path = ".\facial_image\input_0\image_0\face_6\l_eye.jpg"

class_names = ['0 Opened', '1 Closed']
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open(f"{image_path}").convert("RGB")

size = (224, 224)
image = ImageOps.fit(image, size, method=Image.LANCZOS)
image_array = np.asarray(image)
normed_img = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normed_img

prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]


print("Class:", class_name[2:])
# print("Confidence Score:", confidence_score)
