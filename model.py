import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from video frames
def extract_features_from_video(video_path, model, frame_size=(224, 224), frame_step=5):
    cap = cv2.VideoCapture(video_path)
    features_list = []
    success, frame = cap.read()
    frame_count = 0

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
        return None
    return np.array(features_list)

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to make predictions
def predict_deepfake(video_path):
    extracted_features = extract_features_from_video(video_path, feature_extraction_model)
    
    if extracted_features is None:
        return {"error": "Feature extraction failed"}

    lstm_model = create_lstm_model((extracted_features.shape[0], extracted_features.shape[1]))

    reshaped_features = np.expand_dims(extracted_features, axis=0)
    predictions = lstm_model.predict(reshaped_features)

    probability_real = predictions[0][0]
    probability_deepfake = predictions[0][1]

    result = {
        "real_probability": float(probability_real),
        "deepfake_probability": float(probability_deepfake),
        "verdict": "Deepfake" if probability_deepfake > probability_real else "Real"
    }

    return result
