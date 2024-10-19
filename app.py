import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = load_model('resnet50.h5')

# Function to extract frames and preprocess for the model
def extract_frames(video_path, frame_skip=5, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_skip == 0:
            frame_resized = cv2.resize(frame, target_size)
            frame_resized = frame_resized / 255.0  # Normalize the frame
            frames.append(frame_resized)

        frame_count += 1

    cap.release()
    return np.array(frames)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

 
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part in the request"

    file = request.files['file']
    if file.filename == '':
        return "No file selected for uploading"

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded video
        file.save(file_path)

        # Extract frames from the video and predict using the model
        frames = extract_frames(file_path, frame_skip=10)
        predictions = model.predict(frames)

        # Compute average prediction
        avg_prediction = np.mean(predictions)
        classification = 'Real' if avg_prediction >= 0.5 else 'Fake'

        return render_template('result.html', 
                               label=classification, 
                               probability=avg_prediction if classification == 'Real' else (1 - avg_prediction),
                               confidence_deepfake=(1 - avg_prediction),
                               video_path=file_path)
        # return  f"Video classified as: {classification} (Confidence: {avg_prediction:.4f})"

@app.route('/impactofdeepfakes')
def impactofdeepfakes():
    return render_template('impactofdeepfakes.html')

@app.route('/recentcases')
def recentcases():
    return render_template('recentcases.html')
if __name__ == "__main__":
    app.run(debug=True)