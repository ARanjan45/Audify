import sys
import pandas as pd
import modal
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
import numpy as np
import torch.nn as nn
import torchaudio.transforms as T
from model import AudioCNN
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

app = modal.App("audio-cnn")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
    .run_commands([
        "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
    ])
    .add_local_python_source("model")
)

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None, target_length=220500):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.target_length = target_length  # 5 seconds at 44100 Hz

        # Use folds 1-4 for training, fold 5 for validation
        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] <= 4]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata["category"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to target length
        if waveform.shape[1] < self.target_length:
            # Pad with zeros
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.target_length:
            # Random crop during training, center crop during validation
            if self.transform is not None:
                start = np.random.randint(0, waveform.shape[1] - self.target_length)
            else:
                start = (waveform.shape[1] - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, row["label"]


def mixup_data(x, y):
    lam = np.random.beta(0.4, 0.4)  # Changed from 0.2 to 0.4 for less aggressive mixup
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume, "/models": model_volume},
    timeout=60 * 60 * 3,
)
def train():
    from datetime import datetime

    writer = SummaryWriter(f"/models/tensorboard_logs/run_{datetime.now():%Y%m%d_%H%M%S}")

    esc50_dir = Path("/opt/esc50-data")

    # Fixed parameters to avoid the warning
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100, 
            n_mels=128, 
            n_fft=2048,  # Increased from 1024
            hop_length=512,
            f_min=0,
            f_max=22050  # Changed from default to nyquist frequency
        ),
        T.AmplitudeToDB(top_db=80),
        T.FrequencyMasking(freq_mask_param=15),  # Reduced from 30
        T.TimeMasking(time_mask_param=35),  # Reduced from 80
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100, 
            n_mels=128, 
            n_fft=2048,
            hop_length=512,
            f_min=0,
            f_max=22050
        ),
        T.AmplitudeToDB(top_db=80),
    )

    train_dataset = ESC50Dataset(
        esc50_dir, esc50_dir / "meta" / "esc50.csv", "train", train_transform
    )
    val_dataset = ESC50Dataset(
        esc50_dir, esc50_dir / "meta" / "esc50.csv", "val", val_transform
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AudioCNN(num_classes=len(train_dataset.classes)).to(device)

    print("Calculating normalization statistics...")
    with torch.no_grad():
        sample_batch, _ = next(iter(train_loader))
        mean = sample_batch.mean().item()
        std = sample_batch.std().item()
    print(f"Normalization - Mean: {mean:.4f}, Std: {std:.4f}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)  

    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,  
        epochs=150,  
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  
        anneal_strategy='cos'
    )

    best_accuracy = 0.0
    num_epochs = 150
    patience = 20
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            
            data = (data - mean) / (std + 1e-8)

            if np.random.rand() > 0.5:
                data, y_a, y_b, lam = mixup_data(data, target)
                outputs = model(data)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(data)
                loss = criterion(outputs, target)
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        class_correct = [0] * len(train_dataset.classes)
        class_total = [0] * len(train_dataset.classes)

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                data = (data - mean) / (std + 1e-8)
                
                outputs = model(data)
                loss = criterion(outputs, target)

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                val_loss += loss.item()
                
                for i in range(len(target)):
                    label = target[i].item()
                    class_correct[label] += (predicted[i] == target[i]).item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"\nEpoch {epoch+1}/{num_epochs}\n"
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%\n"
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%\n"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "classes": train_dataset.classes,
                    "mean": mean,
                    "std": std,
                },
                "/models/best_model.pth",
            )
            print(f"✓ New best model saved: {accuracy:.2f}%")
            
            print("\nPer-class accuracy:")
            for i, class_name in enumerate(train_dataset.classes[:10]):  
                if class_total[i] > 0:
                    acc = 100 * class_correct[i] / class_total[i]
                    print(f"  {class_name}: {acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        model_volume.commit()

    writer.close()
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")
    return best_accuracy


@app.local_entrypoint()
def main():
    train.remote()