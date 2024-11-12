const video = document.getElementById("videoElement");
const reactionDisplay = document.getElementById("reactionDisplay");
const emojiImage = document.getElementById("emojiImage");

// Start video stream from webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(error => console.error("Camera error:", error));

// Capture frame from the video feed and send it to the backend for prediction
async function captureAndPredict() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64 image
    const imageData = canvas.toDataURL("image/jpeg").split(",")[1];

    // Send the base64 image to the backend for emotion prediction
    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageData })
    });

    // Process the prediction response from the server
    const result = await response.json();

    if (result.emotion) {
        // Update the displayed emotion and reaction
        reactionDisplay.innerText = `Detected Emotion: ${result.emotion}`;
        emojiImage.src = `static/emojis/${result.emoji}`;
    }
}

// Capture a frame every second and send it for prediction
setInterval(captureAndPredict, 1000);
