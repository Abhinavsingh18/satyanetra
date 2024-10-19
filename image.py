from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(_name_)

# Load the pretrained model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define paths for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Feature extraction
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

# Euclidean distance calculation
def euclidean_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

# Main route to upload and process images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        real_image = request.files['real_image']
        fake_image = request.files['fake_image']

        real_path = os.path.join(app.config['UPLOAD_FOLDER'], real_image.filename)
        fake_path = os.path.join(app.config['UPLOAD_FOLDER'], fake_image.filename)
        
        real_image.save(real_path)
        fake_image.save(fake_path)
        
        real_features = extract_features(real_path)
        fake_features = extract_features(fake_path)
        distance = euclidean_distance(real_features, fake_features)
        
        threshold = 0.5
        if distance > threshold:
            result = "The second image is likely a deepfake."
        else:
            result = "The second image is likely real."

        return render_template('result2.html', result=result, real_image=real_image.filename, fake_image=fake_image.filename)
    return render_template('index2.html')

if _name_ == "_main_":
    app.run(debug=True)