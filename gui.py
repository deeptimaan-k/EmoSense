import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Load the trained model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load the model weights
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

# Emotion and emoji dictionaries
emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
reaction_dict = {
    0: "Take a deep breath!", 1: "Stay calm.", 2: "afraid.", 
    3: "You're happy!", 4: "Stay neutral.", 5: "Feeling sad?", 
    6: "Wow! That’s surprising!"
}
emoji_dist = {
    0: "./emojis/angry.png", 1: "./emojis/disgusted.png", 2: "./emojis/fearful.png",
    3: "./emojis/happy.png", 4: "./emojis/neutral.png", 5: "./emojis/sad.png", 6: "./emojis/surprised.png"
}

# Initialize the Tkinter window
window = tk.Tk()
window.title("Emotion Detection")
window.geometry("900x600")
window.configure(bg="#f0f0f0")

# Create a title heading
title_label = Label(window, text="EmoSense", font=('Helvetica', 20, 'bold'), bg="#34495e", fg="white")
title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky='nsew')

# Create frames for left and right side layout with styling
left_frame = Frame(window, width=600, height=500, bg="#2c3e50", relief=RIDGE, bd=5)
left_frame.grid(row=1, column=0, padx=10, pady=10)

right_frame = Frame(window, width=300, height=500, bg="#2c3e50", relief=RIDGE, bd=5)
right_frame.grid(row=1, column=1, padx=10, pady=10)

# Heading for emoji section
heading_label = Label(right_frame, text="Detected Emotion", font=('Arial', 16, 'bold'), bg="#2c3e50", fg="white")
heading_label.pack(pady=10)

# Initialize Label to display the emoji
emoji_label = Label(right_frame, bg="#2c3e50")
emoji_label.pack(pady=20)

# Initialize Label to display the reaction
reaction_label = Label(right_frame, text="", font=('Arial', 14), bg="#2c3e50", fg="white")
reaction_label.pack(pady=20)

# Initialize Label to display the frame from video
video_label = Label(left_frame, bg="#2c3e50")
video_label.pack()

# Start the video feed
cap1 = cv2.VideoCapture(0)

def show_vid():
    if not cap1.isOpened():
        print("Can't open the camera")
        return
    
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))

    # Load the Haar Cascade for face detection from OpenCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    bounding_box = cv2.CascadeClassifier(cascade_path)

    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        # Get the corresponding emoji and reaction message
        emoji_path = emoji_dist.get(maxindex, "./emojis/neutral.png")
        reaction_text = reaction_dict.get(maxindex, "Stay positive!")
        
        # Load and display the emoji image
        emoji_img = Image.open(emoji_path)
        emoji_img = emoji_img.resize((150, 150))
        emoji_img = ImageTk.PhotoImage(emoji_img)
        
        # Update the emoji and reaction labels
        emoji_label.config(image=emoji_img)
        emoji_label.image = emoji_img  # Keep a reference to avoid garbage collection
        reaction_label.config(text=reaction_text)

    # Convert the frame to RGB format for Tkinter compatibility
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame1)
    img = img.resize((600, 500))
    img = ImageTk.PhotoImage(img)

    # Update the video label with the frame
    video_label.config(image=img)
    video_label.image = img  # Keep a reference to avoid garbage collection

    # Update the video feed every 10ms
    window.after(10, show_vid)

# Start video loop
show_vid()

# Run the Tkinter event loop
window.mainloop()

# Release the video capture when the window is closed
cap1.release()
