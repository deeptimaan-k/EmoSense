from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the emotion detection model
emotion_model = load_model('emotion_model.h5')

# Emotion and emoji dictionaries
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
reaction_dict = {
    0: "Take a deep breath!", 1: "Stay calm.", 2: "It’s okay to be afraid.", 
    3: "You're happy!", 4: "Stay neutral.", 5: "Feeling sad? Reach out.", 
    6: "Wow! That’s surprising!"
}
emoji_dist = {
    0: "angry.png", 1: "disgusted.png", 2: "fearful.png",
    3: "happy.png", 4: "neutral.png", 5: "sad.png", 6: "surprised.png"
}

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the webcam feed and emotion prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get image from request
        img_data = request.json['image']
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
        img = img.convert('L')
        img = img.resize((48, 48))
        img = np.array(img).reshape(1, 48, 48, 1) / 255.0

        # Predict the emotion
        prediction = emotion_model.predict(img)
        maxindex = int(np.argmax(prediction))

        # Get emoji path and reaction message
        emoji_path = emoji_dist.get(maxindex, "neutral.png")
        reaction_text = reaction_dict.get(maxindex, "Stay positive!")

        return jsonify({
            'emotion': emotion_dict[maxindex],
            'emoji': emoji_path,
            'reaction': reaction_text
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
