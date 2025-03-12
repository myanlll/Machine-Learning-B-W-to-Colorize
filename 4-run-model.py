import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import os
import random
import shutil
from skimage import exposure
import argparse

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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        d1 = self.dec1(e6)
        d2 = self.dec2(torch.cat([d1, e5], dim=1))
        d3 = self.dec3(torch.cat([d2, e4], dim=1))
        d4 = self.dec4(torch.cat([d3, e3], dim=1))
        d5 = self.dec5(torch.cat([d4, e2], dim=1))
        out = self.dec6(torch.cat([d5, e1], dim=1))
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

parser = argparse.ArgumentParser(description="Colorize black and white images using U-Net")
parser.add_argument('--model_path', type=str, default='models/generator.pth')
parser.add_argument('--input_dirs', type=str, nargs='+', default=['data/processed_train', 'data/processed_val'])
parser.add_argument('--output_dir', type=str, default='output')
args = parser.parse_args()

try:
    model = UNet().to(device)
    model_path = os.path.join(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

transform = transforms.ToTensor()
output_dir = os.path.join(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

image_list = []
for img_dir in args.input_dirs:
    if os.path.exists(img_dir):
        image_list.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
if not image_list:
    raise FileNotFoundError(f"No .jpg files found in {args.input_dirs}")

num_images = random.randint(3, 5)
selected_images = random.sample(image_list, min(num_images, len(image_list)))
print(f"Testing with {len(selected_images)} images")

for idx, img_path in enumerate(selected_images):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if width != 512 or height != 512:
        raise ValueError(f"Image {img_path} must be 512x512, got {width}x{height}")

    gray_img = img.convert('L')
    gray_path = os.path.join(output_dir, f'{idx + 1}_gray.jpg')
    gray_img.save(gray_path)
    print(f"Saved gray: {gray_path}")

    img_tensor = transform(img).unsqueeze(0).to(device)
    img_lab = rgb2lab(img_tensor.permute(0, 2, 3, 1).cpu().numpy()[0])
    img_l = torch.tensor(img_lab[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 100.0
    true_rgb = img_tensor.permute(0, 2, 3, 1).cpu().numpy()[0]

    with torch.no_grad():
        ab_pred = model(img_l)
        ab_pred = ab_pred * 128.0
        ab_pred = ab_pred.cpu().numpy()
        ab_pred_mean = np.mean(ab_pred, axis=(1, 2), keepdims=True)
        ab_pred = ab_pred - ab_pred_mean
        a_channel = ab_pred[0, :, :, 0]
        b_channel = ab_pred[0, :, :, 1]
        sepia_mask = (a_channel > 3) & (b_channel > 3)
        a_channel[sepia_mask] = a_channel[sepia_mask] * 0.02
        b_channel[sepia_mask] = b_channel[sepia_mask] * 0.02
        a_channel = a_channel - 10
        b_channel = b_channel - 20
        for channel in range(2):
            ab_pred[0, :, :, channel] = exposure.equalize_hist(ab_pred[0, :, :, channel])
        ab_pred = ab_pred * 2.0
        ab_pred = np.clip(ab_pred, -128, 128)
        ab_pred = torch.tensor(ab_pred).to(device)
        img_l = img_l * 100.0
        result = torch.cat([img_l, ab_pred], dim=1).permute(0, 2, 3, 1).cpu().numpy()[0]
        predicted_rgb = lab2rgb(result)

    predicted_img = Image.fromarray((predicted_rgb * 255).astype(np.uint8))
    enhancer = ImageEnhance.Color(predicted_img)
    predicted_img_enhanced = enhancer.enhance(2.0)
    predicted_path = os.path.join(output_dir, f'{idx + 1}_predicted.jpg')
    predicted_img_enhanced.save(predicted_path)
    print(f"Saved predicted: {predicted_path}")

    true_img = Image.fromarray((true_rgb * 255).astype(np.uint8))
    true_path = os.path.join(output_dir, f'{idx + 1}_true.jpg')
    true_img.save(true_path)
    print(f"Saved true: {true_path}")

    collage_width = 512 * 3
    collage_height = 512
    collage = Image.new('RGB', (collage_width, collage_height))
    gray_rgb = gray_img.convert('RGB')
    collage.paste(gray_rgb, (0, 0))
    collage.paste(predicted_img_enhanced, (512, 0))
    collage.paste(true_img, (1024, 0))
    collage_path = os.path.join(output_dir, f'{idx + 1}_collage.jpg')
    collage.save(collage_path)
    print(f"Saved collage: {collage_path}")

print("Testing done!")