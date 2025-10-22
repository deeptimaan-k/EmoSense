import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import platform

# Detect OS for better compatibility
IS_MAC = platform.system() == 'Darwin'

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
try:
    emotion_model.load_weights('emotion_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

cv2.ocl.setUseOpenCL(False)

# Emotion and emoji dictionaries
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

reaction_dict = {
    0: "Take a deep breath and relax!", 
    1: "Something's bothering you?", 
    2: "Don't worry, everything will be fine!", 
    3: "Keep smiling! You're doing great!", 
    4: "You seem calm and composed.", 
    5: "Cheer up! Better days are ahead.", 
    6: "Wow! That's surprising!"
}

emoji_dist = {
    0: "./emojis/angry.png", 
    1: "./emojis/disgusted.png", 
    2: "./emojis/fearful.png",
    3: "./emojis/happy.png", 
    4: "./emojis/neutral.png", 
    5: "./emojis/sad.png", 
    6: "./emojis/surprised.png"
}

# Color scheme
BG_COLOR = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT_COLOR = "#0f3460"
TEXT_COLOR = "#eaeaea"
HIGHLIGHT_COLOR = "#e94560"

# Initialize the Tkinter window
window = tk.Tk()
window.title("EmoSense - Real-Time Emotion Detection")

# Window dimensions
window_width = 1100
window_height = 650

# Center window on screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = int((screen_width - window_width) / 2)
y = int((screen_height - window_height) / 2)
window.geometry(f"{window_width}x{window_height}+{x}+{y}")
window.configure(bg=BG_COLOR)

# Prevent window resize for consistent UI
window.resizable(False, False)

# Configure grid weights for responsive layout
window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=2)
window.grid_columnconfigure(1, weight=1)

# Create a title heading with better styling
title_frame = Frame(window, bg=ACCENT_COLOR, height=70)
title_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=0, pady=0)
title_frame.grid_propagate(False)

title_label = Label(
    title_frame, 
    text="😃 EmoSense", 
    font=('Helvetica', 28, 'bold'), 
    bg=ACCENT_COLOR, 
    fg=TEXT_COLOR
)
title_label.pack(expand=True)

subtitle_label = Label(
    title_frame, 
    text="Real-Time Emotion Detection System", 
    font=('Helvetica', 12), 
    bg=ACCENT_COLOR, 
    fg=TEXT_COLOR
)
subtitle_label.pack()

# Create frames for left and right side layout
left_frame = Frame(window, bg=CARD_BG, relief=FLAT, bd=0)
left_frame.grid(row=1, column=0, padx=15, pady=15, sticky='nsew')

right_frame = Frame(window, bg=CARD_BG, relief=FLAT, bd=0)
right_frame.grid(row=1, column=1, padx=(0, 15), pady=15, sticky='nsew')

# Video frame container
video_container = Frame(left_frame, bg=ACCENT_COLOR, relief=FLAT, bd=2)
video_container.pack(padx=10, pady=10, fill=BOTH, expand=True)

video_label = Label(video_container, bg="#000000")
video_label.pack(padx=5, pady=5)

# Status label
status_label = Label(
    left_frame, 
    text="📹 Camera Active", 
    font=('Arial', 11), 
    bg=CARD_BG, 
    fg="#4ecca3"
)
status_label.pack(pady=5)

# Right panel - Emotion Display
emotion_title = Label(
    right_frame, 
    text="Detected Emotion", 
    font=('Helvetica', 20, 'bold'), 
    bg=CARD_BG, 
    fg=TEXT_COLOR
)
emotion_title.pack(pady=(20, 10))

# Emotion name label
emotion_name_label = Label(
    right_frame, 
    text="No Face Detected", 
    font=('Arial', 24, 'bold'), 
    bg=CARD_BG, 
    fg=HIGHLIGHT_COLOR
)
emotion_name_label.pack(pady=10)

# Emoji container
emoji_container = Frame(right_frame, bg=ACCENT_COLOR, relief=FLAT, bd=2)
emoji_container.pack(pady=15)

emoji_label = Label(emoji_container, bg=ACCENT_COLOR)
emoji_label.pack(padx=20, pady=20)

# Confidence label
confidence_label = Label(
    right_frame, 
    text="Confidence: --", 
    font=('Arial', 14), 
    bg=CARD_BG, 
    fg=TEXT_COLOR
)
confidence_label.pack(pady=10)

# Reaction container
reaction_container = Frame(right_frame, bg=ACCENT_COLOR, relief=FLAT, bd=0)
reaction_container.pack(pady=15, padx=20, fill=X)

reaction_label = Label(
    reaction_container, 
    text="Waiting for detection...", 
    font=('Arial', 13), 
    bg=ACCENT_COLOR, 
    fg=TEXT_COLOR,
    wraplength=250,
    justify=CENTER
)
reaction_label.pack(pady=15, padx=15)

# Initialize video capture with better error handling
cap1 = None
try:
    cap1 = cv2.VideoCapture(0)
    if IS_MAC:
        # macOS specific settings for better performance
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap1.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap1.isOpened():
        status_label.config(text="⚠️ Camera Not Available", fg=HIGHLIGHT_COLOR)
        print("Error: Could not open camera")
except Exception as e:
    print(f"Camera initialization error: {e}")
    status_label.config(text="⚠️ Camera Error", fg=HIGHLIGHT_COLOR)

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Smoothing variables for stable predictions
prediction_history = []
history_size = 5

def get_smoothed_prediction(current_pred):
    """Apply temporal smoothing to reduce jitter"""
    prediction_history.append(current_pred)
    if len(prediction_history) > history_size:
        prediction_history.pop(0)
    
    # Average predictions
    avg_pred = np.mean(prediction_history, axis=0)
    return int(np.argmax(avg_pred))

def show_vid():
    if cap1 is None or not cap1.isOpened():
        window.after(100, show_vid)
        return
    
    flag1, frame1 = cap1.read()
    
    if not flag1:
        status_label.config(text="⚠️ Camera Read Error", fg=HIGHLIGHT_COLOR)
        window.after(100, show_vid)
        return
    
    # Resize frame for display
    display_width = 680
    display_height = 510
    frame1 = cv2.resize(frame1, (display_width, display_height))
    
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with improved parameters
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_detected = False
    
    for (x, y, w, h) in faces:
        face_detected = True
        
        # Draw rectangle around face
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 150), 2)
        
        # Extract face ROI
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        try:
            # Preprocess for model
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype('float32') / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=0)
            cropped_img = np.expand_dims(cropped_img, axis=-1)
            
            # Predict emotion
            prediction = emotion_model.predict(cropped_img, verbose=0)
            
            # Apply smoothing
            maxindex = get_smoothed_prediction(prediction[0])
            confidence = float(prediction[0][maxindex])
            
            # Get emotion details
            emotion_text = emotion_dict.get(maxindex, "Unknown")
            reaction_text = reaction_dict.get(maxindex, "Stay positive!")
            emoji_path = emoji_dist.get(maxindex, "./emojis/neutral.png")
            
            # Update emotion name
            emotion_name_label.config(text=emotion_text, fg=HIGHLIGHT_COLOR)
            
            # Update confidence
            confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
            
            # Update reaction
            reaction_label.config(text=reaction_text)
            
            # Load and display emoji
            try:
                emoji_img = Image.open(emoji_path)
                emoji_img = emoji_img.resize((180, 180), Image.Resampling.LANCZOS)
                emoji_photo = ImageTk.PhotoImage(emoji_img)
                emoji_label.config(image=emoji_photo)
                emoji_label.image = emoji_photo
            except Exception as e:
                print(f"Error loading emoji: {e}")
                emoji_label.config(text="😊", font=('Arial', 80))
            
            # Draw emotion label on frame
            label_bg_color = (0, 200, 100)
            cv2.rectangle(frame1, (x, y-40), (x+w, y), label_bg_color, -1)
            cv2.putText(
                frame1, 
                emotion_text, 
                (x+5, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (255, 255, 255), 
                2
            )
            
        except Exception as e:
            print(f"Prediction error: {e}")
    
    if not face_detected:
        emotion_name_label.config(text="No Face Detected", fg=TEXT_COLOR)
        confidence_label.config(text="Confidence: --")
        reaction_label.config(text="Please position your face in the camera")
        prediction_history.clear()
    
    # Convert frame to RGB for Tkinter
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame1)
    img_tk = ImageTk.PhotoImage(img)
    
    # Update video label
    video_label.config(image=img_tk)
    video_label.image = img_tk
    
    # Schedule next frame (adjust delay for performance)
    delay = 15 if IS_MAC else 10
    window.after(delay, show_vid)

def on_closing():
    """Cleanup when window is closed"""
    if cap1 is not None:
        cap1.release()
    cv2.destroyAllWindows()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

# Start video loop
show_vid()

# Run the Tkinter event loop
window.mainloop()