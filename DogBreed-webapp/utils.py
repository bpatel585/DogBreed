import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
from glob import glob


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


labels = {0:'beagle', 1:'chihuahua', 2:'doberman', 3:'french_bulldog', 4:'golden_retriever', 5:'malamute', 6:'pug', 7:'saint_bernard', 
          8:'scottish_deerhound', 9:'tibetan_mastiff'}



def pipeline_model(path):
    img = image.load_img(path,target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)
    max_preds = []
    pred = pred[0]
    for i in range(5):
        name = labels[pred.argmax()]
        per = round(np.amax(pred)*100,2)
        max_preds.append([name,per])
        ele = pred.argmax()
        pred = np.delete(pred,ele)

    paths = glob('static/uploads/*')
    if len(paths)>5:
        for path in paths[:4]:
            os.remove(path)
    return max_preds
