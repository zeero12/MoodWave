import cv2
import numpy as np
from keras.models import model_from_json
import time
import csv
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load the emotion model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Load the sample song dataset
with open('sample_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    song_dataset = list(reader)

# Define the emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Spotify credentials
SPOTIPY_CLIENT_ID = '67a8269997ba4508bb60ac18d4bb239e'
SPOTIPY_CLIENT_SECRET = 'cc44191c47074060a3e583e1019fe702'
SPOTIPY_REDIRECT_URI = 'http://localhost:3000/api/auth/callback/spotify'

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope="user-library-read user-read-playback-state"))

# Define emotion mapping to Spotify genres
emotion_mapping = {
    "happy": ["pop", "upbeat", "dance"],
    "sad": ["acoustic", "piano", "sad"],
    "angry": ["rock", "metal", "rap"],
    "disgusted": ["chill", "ambient", "jazz"],
    "fearful": ["electronic", "party", "hip-hop"],
    "neutral": ["ambient", "piano", "classical"],
    "surprised": ["ambient", "instrumental", "folk"],
}

# Function to get Spotify recommendations based on emotion
def get_spotify_recommendations(emotion):
    if emotion in emotion_mapping:
        genres = emotion_mapping[emotion]
        recommendations = sp.recommendations(seed_genres=genres, limit=10)
        return [f"{track['name']} by {track['artists'][0]['name']}" for track in recommendations['tracks']]
    else:
        return ["Sorry, I don't have recommendations for that mood."]

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Provide the full path for the Haar Cascade Classifier file
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Introduce a sleep interval in seconds
sleep_interval = 1.2

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces available on the camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # List to store emotion predictions for the current frame
    frame_emotion_predictions = []

    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]
        frame_emotion_predictions.append(emotion_label)

        # Display emotion label on the frame
        cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Perform Spotify recommendation if emotion is detected
    for idx, emotion in enumerate(frame_emotion_predictions):
        if emotion:
            emotion = emotion.lower()
            recommendations = get_spotify_recommendations(emotion)
            y_offset = 50 + idx * 30
            for i, recommendation in enumerate(recommendations):
                cv2.putText(frame, recommendation, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    time.sleep(sleep_interval)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
