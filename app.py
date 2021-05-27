
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import cv2
from PIL import Image
import matplotlib.pyplot as plt
app = Flask(__name__)
IMG_SIZE = 50
model = tf.keras.models.load_model("model.h5")
labels = ["NORMAL", "PNEUMONIA"]
UPLOAD_FOLDER = os.path.join('static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods = ["GET","POST"])
def getimage():
    if request.method == "POST":
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        filename = secure_filename(file.filename)
        filepath=os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        prediction = model.predict([prepare(filepath)])
        output = labels[int(prediction[0])]
        return render_template('index.html',filen="uploads/"+filename,prediction=output)

app.run(debug=True)
