import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths for training and validation
train_dir = 'data/train'
val_dir = 'data/test'

# Data generators for training and validation sets with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Resize images to 48x48
    batch_size=64,
    color_mode="grayscale",  # Convert images to grayscale
    class_mode='categorical'  # Categorical labels for emotions
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Define the model architecture
emotion_model = Sequential()

# Add convolutional and pooling layers with dropout for regularization
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# Flatten the output for the dense layers
emotion_model.add(Flatten())

# Fully connected layers with dropout for regularization
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))

# Output layer for 7 emotion classes with softmax activation
emotion_model.add(Dense(7, activation='softmax'))

# Compile the model
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# Train the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,  # Ensure this is correctly calculated
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64  # Ensure this is also correctly calculated
)

# Save the entire model (architecture + weights)
emotion_model.save('emotion_model.h5')  # Saves the entire model

# Optionally, save just the weights with a proper name (e.g., 'emotion_model_weights.h5')
emotion_model.save_weights('emotion_model.weights.h5')  # Save weights only

# Load the Haar Cascade file from OpenCV for face detection
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
