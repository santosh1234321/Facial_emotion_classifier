# 🎭 Real-Time Emotion Detection System

## 📝 Project Overview
This is a deep learning-based application designed to detect human facial expressions in real-time through a webcam. By utilizing **Transfer Learning** with the **MobileNetV2** architecture, the system achieves a balance of high accuracy and efficient performance suitable for edge devices.

## Note: still a work in progress. 
---

## 🚀 Key Features

* **Real-time Detection**: 
    * Utilizes OpenCV to process live video feeds at high frames per second (FPS).
* **Robust Feature Extraction**: 
    * Leverages a MobileNetV2 model pre-trained on the ImageNet dataset for superior pattern recognition.
* **Comprehensive Emotion Coverage**: 
    * Classifies seven distinct emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
* **Optimized Training**: 
    * Implements class weights during the training phase to handle dataset imbalance and prevent model bias.

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **Deep Learning Framework**: TensorFlow / Keras
* **Computer Vision Library**: OpenCV
* **Infrastructure**: 
    * **Training**: Google Colab (utilizing T4 GPU acceleration)
    * **Inference**: Local Windows environment

---

## 📈 Model Architecture & Performance

The model was trained for **20 epochs** using the **FER-2013 dataset**.

### Architecture Layers
* **Base**: MobileNetV2 (Pre-trained)
* **Pooling**: Global Average Pooling 2D
* **Dense Layer**: 128 units with ReLU activation
* **Regularization**: Dropout layer (0.3) to prevent overfitting

### Training Configuration
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Input Dimensions**: 96x96 pixels in RGB color space

---

## 📂 Project Structure

* `dataset/`: Contains the FER-2013 image data organized into `train` and `test` subfolders.
* `model/`: Dedicated directory for the saved `emotion_model.h5` file.
* `train.py`: The training pipeline optimized for Google Colab environments.
* `run.py`: The local inference script for real-time webcam detection.
* `README.md`: Project documentation and setup guide.

---

## 🔧 Installation & Setup

### 1. Training Phase (Cloud)
* Upload the dataset archive to your Google Colab instance.
* Ensure the runtime type is set to **T4 GPU**.
* Execute `train.py` to generate the weights.
* Download the final `emotion_model.h5` to your local machine.

### 2. Local Environment Setup (Windows)
* Clone the repository to your local directory.
* Install the required dependencies via terminal:
    ```bash
    pip install tensorflow opencv-python numpy
    ```
* Place the `emotion_model.h5` file into the `model/` folder.

### 3. Running the Application
* Open your terminal or command prompt.
* Navigate to the project root and execute:
    ```bash
    python run.py
    ```

