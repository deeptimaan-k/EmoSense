import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
emotion_model = load_model('emotion_model.h5')

# Define the emotion dictionary to map index to emotion
emotion_dict = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Neutral'
}

# Load the Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
bounding_box = cv2.CascadeClassifier(cascade_path)

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Loop over the faces detected
    for (x, y, w, h) in num_faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for emotion detection
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion for the detected face
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))  # Get the index with the highest confidence
        emotion_label = emotion_dict[max_index]  # Get the corresponding emotion label

        # Display the predicted emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame with the detected face(s) and emotion(s)
    cv2.imshow('Emotion Detector', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
