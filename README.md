🎧 Audify — Audio Classification & CNN Visualization

Audify is an interactive AI-powered audio classification system that allows users to upload audio files and visualize how a Convolutional Neural Network (CNN) interprets sound at each layer. The project focuses on model interpretability, showing spectrograms, waveforms, and internal feature maps that lead to final predictions.

🚀 Features

🎵 Upload WAV audio files for real-time inference

📊 Audio waveform and Mel-spectrogram visualization

🧠 CNN-based audio classification

🔍 Layer-wise convolutional feature map visualization

📈 Confidence-based prediction ranking

🖥️ Clean, interactive UI for model introspection

🧠 Model Overview

Input audio is converted into a Mel-spectrogram

Spectrograms are passed through a multi-layer CNN

Intermediate convolutional outputs are captured and visualized

Final softmax layer outputs class probabilities

This approach helps understand what the model learns at each stage, not just the final prediction.

📷 Application Screenshots
Audio Upload Interface
<img src="assets/audify-upload.png" alt="Audio Upload" width="800"/>
Classification Results
<img src="assets/classification-results.png" alt="Classification Results" width="800"/>
Spectrogram & Waveform Visualization
<img src="assets/spectrogram-waveform.png" alt="Spectrogram and Waveform" width="800"/>
CNN Feature Map Visualization
<img src="assets/feature-maps.png" alt="Feature Maps" width="800"/>
🛠️ Tech Stack

Python

PyTorch

Librosa (audio processing)

NumPy

Matplotlib

FastAPI / Flask (inference backend)

React / Next.js (frontend UI)

⚙️ How It Works

User uploads an audio file (.wav)

Audio is normalized and converted to a Mel-spectrogram

Spectrogram is fed into a trained CNN model

Model outputs:

Predicted class probabilities

Intermediate feature maps from each convolutional layer

Results are visualized in the UI