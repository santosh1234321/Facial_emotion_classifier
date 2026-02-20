#Real-Time Emotion Detection System
A deep learning-based application that detects human facial expressions in real-time using a webcam. The project utilizes Transfer Learning with a pre-trained MobileNetV2 model to achieve high accuracy and performance.

##🚀 Features
Real-time Detection: Processes live video feed at high FPS using OpenCV.

High Accuracy: Leverages MobileNetV2 pre-trained on ImageNet for robust feature extraction.

Seven Emotion Classes: Detects Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

Class Balancing: Implements class weights during training to prevent model bias.

##🛠️ Tech Stack
Language: Python

Deep Learning: TensorFlow / Keras

Computer Vision: OpenCV

##Environment: Google Colab (Training) and Local Windows (Inference)

##📂 Project Structure
Plaintext
emotion/
├── dataset/            # FER-2013 Dataset (train/test folders)
├── model/              # Contains the trained 'emotion_model.h5'
├── train.py            # Model training script for Google Colab
├── run.py              # Real-time inference script for Local PC
└── README.md           # Documentation

##📈 Model Performance
The model was trained for 20 epochs on the FER-2013 dataset.

Architecture: MobileNetV2 (Base) + GlobalAveragePooling + Dense (128) + Dropout (0.3).

Optimization: Adam Optimizer with Categorical Crossentropy loss.

Input Size: 96x96 pixels (RGB).

##🔧 Installation & Usage
1. Training (Google Colab)
Upload your dataset zip file to Colab.

Run train.py using a T4 GPU runtime.

Download the generated emotion_model.h5.

2. Local Setup
Clone this repository to your desktop.

Install dependencies:

Bash
pip install tensorflow opencv-python numpy
Place the downloaded emotion_model.h5 in the model/ directory.

3. Execution
Run the real-time detection script:

Bash
python run.py
