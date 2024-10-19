# prediction.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Load EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
efficientnet_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_from_video(video_path, model, frame_size=(224, 224), frame_step=5):
    cap = cv2.VideoCapture(video_path)
    features_list = []
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_step == 0:
            resized_frame = cv2.resize(frame, frame_size)
            img_array = np.expand_dims(resized_frame, axis=0)
            img_array = preprocess_input(img_array) 
            
            features = model.predict(img_array)
            features_list.append(features[0])

        frame_count += 1
        success, frame = cap.read()
    
    cap.release()

    if len(features_list) == 0:
        print("Error: No frames processed from the video.")
        return None
    
    return np.array(features_list)

def create_lstm_model(input_shape, hidden_units=512, num_classes=2):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def detect_deepfake(video_path):
    extracted_features = extract_features_from_video(video_path, efficientnet_model)
    
    if extracted_features is None:
        return {'error': 'Feature extraction failed'}

    input_shape = (extracted_features.shape[0], extracted_features.shape[1])
    lstm_model = create_lstm_model(input_shape)

    reshaped_features = np.expand_dims(extracted_features, axis=0)
    predictions = lstm_model.predict(reshaped_features)

    probability_real = predictions[0][0]
    probability_deepfake = predictions[0][1]

    if probability_deepfake > probability_real:
        result = "Deepfake Detected"
    else:
        result = "Real Video"
    
    return {
        'result': result,
        'confidence_real': probability_real,
        'confidence_deepfake': probability_deepfake
    }