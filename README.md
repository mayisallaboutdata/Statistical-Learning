# Real-Time Emotion Detection with Spotify Integration

This repository contains the source code and documentation for our final project in Statistical Learning, developed by Group 21. The project is designed to detect emotions in real time using a webcam and to enhance the user experience by playing emotion-specific Spotify playlists. The system is deployed on a Raspberry Pi using TensorFlow Lite for efficient model inference and the Spotipy library for interacting with the Spotify Web API.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Tools and Technologies](#tools-and-technologies)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Simplification and Conversion](#model-simplification-and-conversion)
- [Raspberry Pi Implementation](#raspberry-pi-implementation)
- [Spotify Integration](#spotify-integration)
- [Results and Future Improvements](#results-and-future-improvements)
- [Authors](#authors)
- [References](#references)
- [License](#license)

## Overview

The goal of this project is to develop a real-time emotion detection system that uses a webcam to capture facial expressions and classifies them into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. Based on the detected emotion, the system plays an appropriate Spotify playlist to enhance the user's mood and overall experience.

## Objectives

- **Real-Time Emotion Detection:** Use a webcam to detect and classify emotions in real time.
- **Spotify Integration:** Map detected emotions to specific Spotify playlists and control playback via the Spotify API.
- **Efficient Deployment:** Implement the solution on a Raspberry Pi for a compact, portable setup.

## Tools and Technologies

- **TensorFlow & TensorFlow Lite:** Used for training the emotion detection model and converting it for efficient deployment on Raspberry Pi.
- **OpenCV:** Utilized for real-time face detection and image processing.
- **Spotipy:** Python library for interfacing with the Spotify Web API.
- **Raspberry Pi:** Hardware platform for deploying the real-time emotion detection system.

## Data Collection and Preprocessing

- **Dataset:** FER-2013 dataset, comprising images categorized into 7 emotion classes.
- **Preprocessing Steps:**
  - Normalization of image pixel values.
  - Conversion of labels to one-hot encoded vectors.
  - Reshaping images to fit the Convolutional Neural Network (CNN) input requirements.

## Model Architecture

The emotion detection model is a CNN that includes:
- Convolutional layers with Batch Normalization and Dropout for feature extraction.
- Dense layers with Dropout for classification.
- A final softmax output layer to predict one of the 7 emotion classes.
- Achieved approximately 65% accuracy on the validation set.

## Model Simplification and Conversion

To optimize the model for deployment on Raspberry Pi:
- The model architecture was simplified to reduce computational complexity.
- The Keras model was converted to TensorFlow Lite format using optimizations such as quantization to decrease model size and improve inference speed.

## Raspberry Pi Implementation

The implementation on Raspberry Pi includes:
- **Software Setup:** Installation of necessary libraries such as OpenCV, numpy, and the TensorFlow Lite runtime.
- **Real-Time Processing:** Capturing live video, detecting faces using OpenCV, preprocessing the images, and performing emotion prediction.
- **Display:** Rendering a video feed with visual indicators showing the detected emotions.
- **Integration:** Communicating with the Spotify API to trigger the playback of emotion-specific playlists.

## Spotify Integration

The project uses the Spotipy library to interact with the Spotify Web API:
- **Authentication:** Configuring the Spotify developer credentials (Client ID, Client Secret, and Redirect URI).
- **Mapping Emotions:** Each detected emotion is associated with a unique Spotify playlist URI.
- **Playback Control:** Automatically starts and manages playlist playback based on the current emotion detected.

## Results and Future Improvements

### Results
- Real-time emotion detection was successfully implemented with an accuracy of approximately 65%.
- The system effectively integrated with Spotify, playing the correct playlists based on detected emotions.
- The Raspberry Pi demonstrated smooth real-time processing and interfacing with both the webcam and Spotify API.

### Future Improvements
- **Model Enhancements:** Explore advanced architectures (e.g., ResNet, EfficientNet) and incorporate data augmentation to improve prediction accuracy.
- **Performance Optimization:** Further optimize the TensorFlow Lite model for faster inference and consider multithreading to parallelize face detection and emotion recognition.
- **Robustness:** Enhance the system’s ability to handle overlapping or ambiguous emotional expressions and refine the emotion-to-playlist mapping.

## Usage

### Prerequisites
- A Raspberry Pi with a connected webcam.
- Python 3.x and the following libraries:
  - OpenCV
  - numpy
  - TensorFlow Lite runtime
  - Spotipy
- A Spotify Developer account with valid Client ID, Client Secret, and Redirect URI.

## Authors

- **Deniz Yilmaz (ID: 2108621)**
- **Onur Ozan Sunger (ID: 2113119)**
- **Umut Altun (ID: 2101934)**
- **Selin Topaloglu (ID: 2311462)**
- **Mayis Atayev (ID: 2104359)**

## References

1. Facial Emotion Recognition Dataset, Kaggle. Available Here 
2. Jones et al., "Deep Learning for Real-Time Image Processing on Embedded Devices," 2020.
3. Smith et al., "Emotion Recognition using Facial Expressions," 2021.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
