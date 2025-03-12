import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import os
import time
import sys
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import psutil
import keyboard

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.enc5 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.enc6 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec5 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec6 = nn.Sequential(nn.ConvTranspose2d(128, 2, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1, e2, e3, e4, e5, e6 = (layer(x) for x, layer in zip([x, self.enc1(x), self.enc2(self.enc1(x)), self.enc3(self.enc2(self.enc1(x))), self.enc4(self.enc3(self.enc2(self.enc1(x)))), self.enc5(self.enc4(self.enc3(self.enc2(self.enc1(x)))))], [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6]))
        d1 = self.dec1(e6)
        d2 = self.dec2(torch.cat([d1, e5], dim=1))
        d3 = self.dec3(torch.cat([d2, e4], dim=1))
        d4 = self.dec4(torch.cat([d3, e3], dim=1))
        d5 = self.dec5(torch.cat([d4, e2], dim=1))
        return self.dec6(torch.cat([d5, e1], dim=1))

class ColorizationDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert('RGB')
        img_tensor = self.transform(img)
        img_lab = rgb2lab(img_tensor.permute(1, 2, 0).numpy())
        img_lab = (img_lab + [0, 128, 128]) / [100, 255, 255]
        return (torch.tensor(img_lab[:, :, 0], dtype=torch.float32).unsqueeze(0), 
                torch.tensor(img_lab[:, :, 1:], dtype=torch.float32).permute(2, 0, 1))

def format_time(seconds, is_estimate=False):
    seconds = int(seconds)
    if is_estimate:
        hours, minutes = seconds // 3600, (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    if seconds < 60:
        return f"{seconds}"
    minutes, seconds = seconds // 60, seconds % 60
    return f"{minutes}min {seconds if seconds else 0}"

def get_system_resources():
    disk_usage = psutil.disk_usage('/').percent
    cpu_usage = psutil.cpu_percent(interval=0.1)
    vram_total = vram_used = vram_percent = 0
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        vram_percent = (vram_used / vram_total) * 100
    return disk_usage, cpu_usage, vram_percent, vram_used, vram_total

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    batch_size = 48
    num_workers = 8
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    start_time = time.time()
    total_batches = len(train_loader) * num_epochs
    processed_batches = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (l_channel, ab_channels) in enumerate(train_loader):
            l_channel, ab_channels = l_channel.to(device), ab_channels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                output = model(l_channel)
                loss = criterion(output, ab_channels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            processed_batches += 1

            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / processed_batches
            remaining_batches = total_batches - processed_batches
            progress = (batch_idx + 1) / len(train_loader) * 100
            total_progress = processed_batches / total_batches * 100
            bar = "█" * int(bar_length := 50 * progress // 100) + " " * (bar_length - int(bar_length))
            sys.stdout.write(f"\rEpoch {epoch+1}/{num_epochs} | [{bar}] {progress:.1f}% | Total: {total_progress:.1f}% | "
                           f"Time: {format_time(elapsed_time)} | ETA: {format_time(avg_time_per_batch * remaining_batches, True)}")
            sys.stdout.flush()

            disk_usage, _, _, vram_used, vram_total = get_system_resources()
            if vram_used < 7.0 and batch_size < 64:
                batch_size += 8
                print(f"\nBoosting batch_size to {batch_size} (VRAM: {vram_used:.1f}/{vram_total:.1f}GB)")
                train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            elif vram_used > 7.5 and batch_size > 8:
                batch_size -= 8
                print(f"\nLowering batch_size to {batch_size} (VRAM: {vram_used:.1f}/{vram_total:.1f}GB)")
                train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            if disk_usage > 94 and num_workers > 1:
                num_workers -= 1
                print(f"\nReducing num_workers to {num_workers} (SSD: {disk_usage:.1f}%)")
                train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            if keyboard.is_pressed('ctrl') and keyboard.is_pressed('ü'):
                print("\nStopped with Ctrl + Ü!")
                return
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for l_channel, ab_channels in val_loader:
                    l_channel, ab_channels = l_channel.to(device), ab_channels.to(device)
                    with autocast('cuda'):
                        output = model(l_channel)
                        loss = criterion(output, ab_channels)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
        else:
            val_loss = float('inf')
            print("\nNo validation data, skipping validation.")

        print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}" if val_loss != float('inf') else 
              f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: N/A")
        scheduler.step(val_loss if val_loss != float('inf') else train_loss)

        if val_loss != float('inf') and val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join('models', 'generator.pth'))
            print("Saved best model!")
        elif val_loss != float('inf'):
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping activated.")
                break

    print(f"\nDone! Took {format_time(time.time() - start_time)}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    model = UNet().to(device)
    checkpoint_path = os.path.join('models', 'generator.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Loaded from {checkpoint_path}")
        start_epoch = checkpoint.get('epoch', int(input("Enter last epoch (e.g., 3): "))) + 1 if isinstance(checkpoint, dict) and 'epoch' in checkpoint else int(input("Enter last epoch (e.g., 3): ")) + 1
        print(f"Starting from epoch {start_epoch}")
    else:
        print("No checkpoint, starting fresh.")
        start_epoch = 0

    while True:
        try:
            total_epochs = int(input("Total epochs (e.g., 25): "))
            if total_epochs >= start_epoch:
                break
        except ValueError:
            print("Enter a valid number.")

    train_dataset = ColorizationDataset(os.path.join('data', 'processed_train'))
    val_dataset = ColorizationDataset(os.path.join('data', 'processed_val'))
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=1)

    print(f"Training from {start_epoch} to {total_epochs}...")
    train_model(model, train_loader, val_loader, total_epochs - start_epoch, device)