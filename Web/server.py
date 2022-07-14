"""
Flask Server with AI model
"""
from flask import Flask, render_template, request
from requests import session

from tensorflow import keras
import numpy as np
import cv2
import runway

app = Flask(__name__)

# Read class names
with open("./runway/class_names.txt", "r") as ins:
  class_names = []
  for line in ins:
    class_names.append(line.rstrip('\n'))

# Load the model
model = keras.models.load_model('./runway/doodleNet-model.h5')
model.summary()

@app.route("/", methods=['GET', 'POST'])
def index():
    # if request.method =='GET':
    #     return render_template('index.html')
    if request.method =='GET':

        # open a local image
        #img = cv2.imread('apple.png')
        img = cv2.imread('runway/apple2.png')
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((28, 28, 1))
        img = (255 - img) / 255
        # predict
        pred = model.predict(np.expand_dims(img, axis=0))[0]
        ind = (-pred).argsort()[:5] # ind is index of classname 5
        latex = [class_names[x] for x in ind] # latex is top 10 classname
        print(latex) # 5개 출력됨.
        return render_template('index.html',names=latex)

if __name__ =='__main__':
    app.run(port=5000, debug=True)
#print('test')