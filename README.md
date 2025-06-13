# UrbanSound8K-MultiSoundDetect 🎧🔍

UrbanSound8K-MultiSoundDetect is a deep learning application that uses a pretrained YAMNet model to classify **multiple environmental sounds** within a single audio mixture. It is designed for tasks involving soundscape analysis, urban noise monitoring, and intelligent audio tagging.

## 🚀 Features

- Supports **multi-label classification** (detects multiple sounds in one clip)
- Based on **YAMNet** pretrained on Google's AudioSet
- Lightweight classifier trained on mixtures of UrbanSound8K samples
- Simple web interface using **Gradio**

## 🧪 Try It

Visit the Hugging Face Space:  
👉 [UrbanSound8K-MultiSoundDetect on HF](https://huggingface.co/spaces/AbdGhordlo/UrbanSound8K-MultiSoundDetect)  
Upload a 4-second `.wav` mixture and see the predicted classes!

## 🛠 Files

- `app.py`: Gradio interface for uploading and classifying audio
- `yamnet_mixture_model.h5`: Trained Keras model
- `class_names.pkl`: Serialized class name list
- `requirements.txt`: Python package requirements
- You can use the wav files in "Test Sounds" for testing the model

## 👨‍💻 Authors

- Abdallah Ghordlo
- Hussein Abdikarim Hussein
- Fatma Özbek

Project for ADA 447 – Introduction to Deep Learning  
