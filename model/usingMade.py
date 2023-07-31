import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
from keras.models import load_model
#model = "E:\DOWNLOAD\eyeblinkdetector.h5"
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model = load_model("E:\DOWNLOAD\eyeblinkdetector.h5" % (start_time))

y_pred = model.predict(x_val/255.)
y_pred_logical = (y_pred > 0.5).astype(np.int)

print ('test acc: %s' % accuracy_score(y_val, y_pred_logical))
cm = confusion_matrix(y_val, y_pred_logical)
sns.heatmap(cm, annot=True)