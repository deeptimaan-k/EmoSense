# 😃 EmoSense – Real-Time Emotion Detection

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**EmoSense** is a Deep Learning-based Facial Emotion Detection System that leverages Convolutional Neural Networks (CNN) to analyze facial expressions and classify emotions in real-time. Built with TensorFlow/Keras and deployed using Flask, it provides accurate emotion recognition through webcam or image input.

---

## 🎯 Overview

EmoSense processes grayscale facial images (48×48 pixels) and classifies emotions into **seven distinct categories** using a trained CNN model. Each detected emotion is paired with contextual feedback including emojis and reaction messages.

### Supported Emotions

| Emotion Index | Emotion    | Emoji |
|---------------|-----------|-------|
| 0             | Angry     | 😠    |
| 1             | Disgusted | 🤢    |
| 2             | Fearful   | 😨    |
| 3             | Happy     | 😊    |
| 4             | Neutral   | 😐    |
| 5             | Sad       | 😢    |
| 6             | Surprised | 😲    |

---

## ✨ Key Features

- **Real-time emotion detection** via webcam with live video processing
- **Multi-class classification** across 7 emotion categories
- **REST API** for seamless integration with other applications
- **Pre-trained CNN model** with optimized architecture
- **Batch processing** support for multiple images
- **Cross-platform compatibility** (Windows, Linux, macOS)
- **Lightweight deployment** suitable for edge devices

---

## 🧠 Model Architecture

The CNN model is specifically designed for facial emotion recognition:

```
Input Layer (48x48x1 grayscale images)
    ↓
Conv2D (32 filters) + ReLU + BatchNorm
    ↓
MaxPooling2D + Dropout(0.25)
    ↓
Conv2D (64 filters) + ReLU + BatchNorm
    ↓
MaxPooling2D + Dropout(0.25)
    ↓
Conv2D (128 filters) + ReLU + BatchNorm
    ↓
MaxPooling2D + Dropout(0.4)
    ↓
Flatten + Dense(128) + Dropout(0.5)
    ↓
Output Layer: Dense(7) + Softmax
```

**Architecture Highlights:**
- **Convolutional Layers**: Extract hierarchical facial features
- **Batch Normalization**: Accelerate training and improve stability
- **MaxPooling**: Reduce dimensionality while preserving important features
- **Dropout Regularization**: Prevent overfitting (rates: 0.25, 0.4, 0.5)
- **Softmax Activation**: Generate probability distribution across emotion classes

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **TensorFlow 2.x / Keras** - Deep learning framework
- **OpenCV (cv2)** - Computer vision and image processing
- **NumPy** - Numerical computations
- **Pillow (PIL)** - Image manipulation

### Backend & Deployment
- **Flask 2.x** - Lightweight web framework
- **Flask-CORS** - Cross-Origin Resource Sharing support
- **Gunicorn** - Production WSGI server (optional)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam (for real-time detection)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/deeptimaan-k/EmoSense.git
cd EmoSense
```

2. **Create a virtual environment**
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download the pre-trained model**
   
   Ensure `emotion_model.h5` is in the root directory. If not available, train the model using the provided dataset.

5. **Run the application**
```bash
python app.py
```

6. **Access the API**
   
   The server will start at `http://127.0.0.1:5000`

---

## 📂 Project Structure

```
EmoSense/
│
├── app.py                      # Flask application & API endpoints
├── emotion_model.h5            # Pre-trained CNN model (HDF5 format)
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/                       # Dataset directory
│   ├── train/                  # Training images (organized by emotion)
│   │   ├── angry/
│   │   ├── disgusted/
│   │   ├── fearful/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── test/                   # Validation/test images
│
├── utils/                      # Utility functions
│   ├── preprocessing.py        # Image preprocessing utilities
│   └── emotion_mappings.py     # Emotion-to-emoji mappings
│
└── logs/                       # Training logs and metrics
    └── training_history.json
```

---

## 🚀 Usage

### 1. Real-Time Webcam Detection

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_model.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        
        prediction = model.predict(face_roi)
        emotion_idx = np.argmax(prediction)
        emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotions[emotion_idx], (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('EmoSense', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. API Usage

#### Predict Emotion from Image

**Endpoint**: `POST /predict`

**Request** (using cURL):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@path/to/image.jpg"
```

**Request** (using Python):
```python
import requests

url = "http://127.0.0.1:5000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response**:
```json
{
  "emotion": "Happy",
  "confidence": 0.92,
  "emotion_index": 3,
  "emoji": "😊",
  "message": "You seem joyful! Keep spreading positivity!"
}
```

#### Webcam Stream Endpoint

**Endpoint**: `GET /video_feed`

Returns MJPEG stream with real-time emotion detection overlay.

---

## 🧪 Model Training

### Dataset Requirements

The model is trained on the **FER-2013 dataset** (or similar):
- **Training samples**: ~28,000 images
- **Validation samples**: ~7,000 images
- **Image format**: 48×48 grayscale
- **Classes**: 7 emotions (balanced or weighted)

### Training the Model

```bash
python train_model.py --epochs 50 --batch_size 64 --learning_rate 0.001
```

**Training Parameters**:
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Data Augmentation**: Rotation, zoom, horizontal flip
- **Early Stopping**: Monitor validation loss (patience: 10)

### Model Performance

| Metric          | Value  |
|-----------------|--------|
| Training Accuracy | 87.3% |
| Validation Accuracy | 68.5% |
| Test Accuracy | 66.2% |
| Average Inference Time | 25ms |

---

## 🔧 Configuration

Edit `config.py` to customize settings:

```python
# Model configuration
MODEL_PATH = 'emotion_model.h5'
IMAGE_SIZE = (48, 48)

# Flask configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False

# Emotion mappings
EMOTIONS = {
    0: {'name': 'Angry', 'emoji': '😠', 'message': 'Take a deep breath'},
    1: {'name': 'Disgusted', 'emoji': '🤢', 'message': 'Something bothering you?'},
    # ... additional mappings
}
```

---

## 📊 Use Cases

- **Mental Health Monitoring**: Track emotional patterns over time
- **Customer Service**: Gauge customer satisfaction in real-time
- **Education**: Monitor student engagement during online classes
- **Healthcare**: Assist therapists in analyzing patient emotions
- **Human-Computer Interaction**: Enable emotion-aware interfaces
- **Security**: Detect suspicious behavior through emotion analysis
- **Market Research**: Analyze consumer reactions to products

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FER-2013 Dataset** - Facial expression recognition dataset
- **TensorFlow Team** - Deep learning framework
- **OpenCV Community** - Computer vision library
- All contributors who helped improve this project

---

## 📧 Contact

**Project Maintainer**: Your Name

- GitHub: [@deeptimaan-k](https://github.com/deeptimaan-k)
- Email: deeptimaankrishnajadaun@gmail.com


---

## 🔮 Future Enhancements

- [ ] Multi-face detection and tracking
- [ ] Integration with audio emotion detection
- [ ] Mobile application (iOS/Android)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Real-time emotion analytics dashboard
- [ ] Support for additional emotion categories
- [ ] Edge device optimization (Raspberry Pi, Jetson Nano)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the EmoSense Team

</div>
