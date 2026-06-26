import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def speak_text(text):
    def run_speech():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run_speech, daemon=True).start()

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

actions = np.array(['hello', 'my', 'name', 'thanks', 'A', 'S', 'H', 'I', 'Q'])

model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 258)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

try:
    model.load_weights('asl_model_filteredaugment.h5') 
    print("Weights loaded successfully.")
except:
    weights_data = np.load('asl_weights_filteredaugment.npy', allow_pickle=True)
    model.set_weights(list(weights_data))
    print("Weights loaded from .npy successfully.")

sequence = []
sentence = []
predictions = []
current_spelling = []
spelling_letters = ['A', 'S', 'H', 'I', 'Q']

threshold = 0.7 
stability_frames = 10 
cooldown_counter = 0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        try:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
        except: continue

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            confidence = res[predicted_idx]
            predictions.append(predicted_idx)

            if len(predictions) >= stability_frames:
                if len(set(predictions[-stability_frames:])) == 1:
                    if confidence > threshold and cooldown_counter == 0:
                        detected_action = actions[predicted_idx]
                        
                        if detected_action in spelling_letters:
                            if not current_spelling or detected_action != current_spelling[-1]:
                                current_spelling.append(detected_action)
                                cooldown_counter = 30
                                
                                if detected_action == 'Q' or len(current_spelling) >= 5:
                                    full_word = "".join(current_spelling)
                                    sentence.append(full_word)
                                    speak_text(full_word)
                                    current_spelling = []
                        
                        else:
                            if not sentence or detected_action != sentence[-1]:
                                sentence.append(detected_action)
                                speak_text(detected_action)
                                cooldown_counter = 25

        if cooldown_counter > 0: cooldown_counter -= 1

        spelling_preview = "".join(current_spelling)
        ui_text = " ".join(sentence).upper()
        if spelling_preview: ui_text += f" ({spelling_preview})"

        cv2.rectangle(image, (0, 0), (640, 45), (245, 117, 16), -1)
        cv2.putText(image, ui_text, (15, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign Language Detection', image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            sentence = []
            current_spelling = []

cap.release()
cv2.destroyAllWindows()
