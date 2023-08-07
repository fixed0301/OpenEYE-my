from keras.models import load_model
import cv2, os

model_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '\\data_processing\\model\\model3.h5'
def classify(facial_img):
    model = load_model(model_path)
    facial_img = cv2.resize(facial_img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)  # 200*200*3 input
    facial_img = facial_img.reshape((-1, 200, 200, 3))
    answer = (model.predict(facial_img) > 0.5).astype("int32")
    if answer == [0]:
        return 'closed'
    else:
        return 'opened'
