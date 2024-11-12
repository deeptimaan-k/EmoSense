# EmoSense

EmoSense is an intelligent sentiment analysis tool designed to analyze text and provide a detailed emotional breakdown. It leverages advanced machine learning algorithms to determine the overall emotional tone of text input, enabling users to gain insights into the sentiments expressed.

## Features
- **Accurate Sentiment Detection**: Analyzes input text to determine emotional tones such as joy, sadness, anger, and more.
- **Detailed Sentiment Scores**: Provides a comprehensive breakdown of the emotional composition.
- **Real-Time Analysis**: Processes input efficiently to deliver quick results.
- **User-Friendly Interface**: Simple and intuitive for easy interaction.

## Technologies Used
- **Programming Language**: Python
- **Frameworks and Libraries**:
  - Flask: For building the web interface.
  - scikit-learn: Utilized for implementing machine learning models.
  - NLTK (Natural Language Toolkit): For natural language processing tasks.
- **Front-End**: HTML/CSS, JavaScript
- **Deployment**: Hosted on a cloud platform for online access.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/deeptimaan-k/EmoSense.git
   cd EmoSense
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   python app.py
   ```
   Access the app at `http://127.0.0.1:5000/`.

## Usage
1. Open the EmoSense application in your browser.
2. Input the text you want to analyze.
3. Click on the 'Analyze' button.
4. View the sentiment analysis results, including a detailed emotional breakdown.

## Project Structure
```
EmoSense/
|-- app.py
|-- requirements.txt
|-- static/
|   |-- css/
|   |-- js/
|-- templates/
|   |-- index.html
|-- models/
|   |-- sentiment_model.pkl
|-- README.md
```

## Contributing
Contributions are welcome! Feel free to submit issues or create pull requests to improve the project.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please reach out to deeptimaankrishnajadaun@gmail.com.

