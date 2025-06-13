import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import pickle

# Load the trained model and class names
model = tf.keras.models.load_model('yamnet_mixture_model.h5')
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Feature extractor
def get_yamnet_embeddings(wav_path, target_sr=16000):
    y, sr = librosa.load(wav_path, sr=target_sr)
    scores, embeddings, spectrogram = yamnet_model(y)
    return embeddings.numpy()

# Prediction function
def predict_mixture(file):
    emb = get_yamnet_embeddings(file)
    avg_emb = np.mean(emb, axis=0)[np.newaxis, :]
    pred = model.predict(avg_emb)[0]

    top_classes = [class_names[i] for i, p in enumerate(pred) if p >= 0.5]
    confidences = {class_names[i]: float(np.round(p, 3)) for i, p in enumerate(pred) if p >= 0.5}

    if not top_classes:
        return "No confident prediction (all < 0.5 threshold)."
    
    return f"Predicted Classes:\n{', '.join(top_classes)}\n\nConfidences:\n{confidences}"

# Gradio UI
interface = gr.Interface(
    fn=predict_mixture,
    inputs=gr.Audio(type='filepath', label="Upload a mixture (.wav)"),
    outputs=gr.Textbox(label="Predicted Sound Classes"),
    title="UrbanSound8K Mixture Classifier (YAMNet + TF)",
    description="Upload a 4-second mixture sample to identify the top sound classes in it."
)

interface.launch()
