import base64
import io
import modal
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from model import AudioCNN
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import librosa


app = modal.App("audio-cnn-inference")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install("libsndfile1")
    .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model")

class AudioProcessor:
    def __init__(self, mean=0.0, std=1.0):
        # Match training parameters EXACTLY
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=44100,
                n_fft=2048,  # Changed to match training
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=22050  # Changed to match training
            ),
            T.AmplitudeToDB(top_db=80)
        )
        self.mean = mean
        self.std = std
        self.target_length = 220500  # 5 seconds at 44100 Hz

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        
        # Pad or trim to target length (match training)
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.target_length:
            # Center crop for inference
            start = (waveform.shape[1] - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
        
        # Generate spectrogram
        spectrogram = self.transform(waveform)
        
        # Normalize using training statistics
        spectrogram = (spectrogram - self.mean) / (self.std + 1e-8)
        
        return spectrogram.unsqueeze(0)

class InferenceRequest(BaseModel):
    audio_data: str

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": model_volume},
    timeout=300,
    container_idle_timeout=120
)
class AudioClassifier:

    @modal.enter()
    def load_model(self):
        print("Loading model on enter...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            "/models/best_model.pth",
            map_location=self.device,
            weights_only=False
        )

        self.classes = checkpoint["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization statistics
        mean = checkpoint.get("mean", 0.0)
        std = checkpoint.get("std", 1.0)
        self.audio_processor = AudioProcessor(mean=mean, std=std)

        print(f"Model loaded successfully.")
        print(f"Normalization - Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"Number of classes: {len(self.classes)}")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        try:
            audio_bytes = base64.b64decode(request.audio_data)
            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32"
            )

            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample to 44100 if needed
            if sample_rate != 44100:
                audio_data = librosa.resample(
                    y=audio_data,
                    orig_sr=sample_rate,
                    target_sr=44100
                )

            # Process audio
            spectrogram = self.audio_processor.process_audio_chunk(audio_data)
            spectrogram = spectrogram.to(self.device)

            with torch.no_grad():
                outputs, feature_maps = self.model(spectrogram, return_feature_maps=True)
                
                # Handle NaN values
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                
                probabilities = torch.softmax(outputs, dim=1)

                # Get top 5 predictions instead of 3
                top5_probs, top5_indices = torch.topk(probabilities[0], min(5, len(self.classes)))

                predictions = [
                    {
                        "class": self.classes[idx.item()],
                        "confidence": float(prob.item())
                    }
                    for prob, idx in zip(top5_probs, top5_indices)
                ]
                
                # Prepare visualization data
                viz_data = {}
                for name, tensor in feature_maps.items():
                    if tensor.dim() == 4:
                        # Aggregate across channels
                        aggregated_tensor = torch.mean(tensor, dim=1)
                        squeezed_tensor = aggregated_tensor.squeeze(0)
                        numpy_array = squeezed_tensor.cpu().numpy()
                        clean_array = np.nan_to_num(numpy_array, nan=0.0, posinf=0.0, neginf=0.0)
                        viz_data[name] = {
                            "shape": list(clean_array.shape),
                            "values": clean_array.tolist()
                        }
                
                # Prepare spectrogram for visualization
                spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
                clean_spectrogram = np.nan_to_num(spectrogram_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Downsample waveform for visualization
                max_samples = 8000
                if len(audio_data) > max_samples:
                    step = len(audio_data) // max_samples
                    waveform_data = audio_data[::step]
                else:
                    waveform_data = audio_data

            return {
                "predictions": predictions,
                "visualizations": viz_data,
                "input_spectrogram": {
                    "shape": list(clean_spectrogram.shape),
                    "values": clean_spectrogram.tolist()
                },
                "waveform": {
                    "values": waveform_data.tolist(),
                    "sample_rate": 44100,
                    "duration": len(audio_data) / 44100
                }
            }
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "predictions": []
            }


@app.local_entrypoint()
def main():
    audio_data, sr = sf.read("chirpingbirds.wav")
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()
    url = server.inference.web_url

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", [])
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get('duration', 0):.2f}s")
    
    print("\nTop Predictions:")
    for pred in result.get("predictions", []):
        print(f"  - {pred['class']}: {pred['confidence']:.2%}")