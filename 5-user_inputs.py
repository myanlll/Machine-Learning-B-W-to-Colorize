import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import os
import argparse
from skimage import exposure

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
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Colorize user-provided images using U-Net")
parser.add_argument('--model_path', type=str, default='models/generator.pth', help='Path to the model file (default: models/generator.pth)')
parser.add_argument('--input_dir', type=str, default='input_user', help='Input directory containing images (default: input_user)')
parser.add_argument('--output_dir', type=str, default='output_user', help='Output directory for results (default: output_user)')
args = parser.parse_args()

try:
    model = UNet().to(device)
    model_path = os.path.join(os.getcwd(), args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded successfully from {model_path}!")
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}")
    exit()
except Exception as e:
    print(f"Error: Failed to load model - {e}")
    exit()

transform = transforms.Compose([
    transforms.ToTensor()
])

output_dir = os.path.join(os.getcwd(), args.output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

image_list = []
full_img_dir = os.path.join(os.getcwd(), args.input_dir)
if os.path.exists(full_img_dir):
    image_list.extend([os.path.join(full_img_dir, f) for f in os.listdir(full_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if not image_list:
    raise FileNotFoundError(f"No image files (.jpg, .jpeg, .png) found in {args.input_dir}!")

print(f"Found {len(image_list)} images for processing: {image_list}")

for idx, img_path in enumerate(image_list):
    try:
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        print(f"Original image {img_path} dimensions: {width}x{height}")

        orig_width, orig_height = width, height

        img_for_model = img.resize((512, 512), Image.Resampling.LANCZOS)
        print(f"Resized image to 512x512 for model input (aspect ratio distorted)")

        img_tensor = transform(img_for_model).unsqueeze(0).to(device)
        print(f"img_tensor shape: {img_tensor.shape}")

        img_lab = rgb2lab(img_tensor.permute(0, 2, 3, 1).cpu().numpy()[0])
        img_l = torch.tensor(img_lab[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 100.0
        print(f"img_lab shape: {img_lab.shape}, img_l shape: {img_l.shape}")

        with torch.no_grad():
            ab_pred = model(img_l)
            print(f"ab_pred shape before scaling: {ab_pred.shape}")
            ab_pred = ab_pred * 128.0
            ab_pred = ab_pred.cpu().numpy()
            print(f"ab_pred shape after scaling: {ab_pred.shape}")

            ab_pred_mean = np.mean(ab_pred, axis=(1, 2), keepdims=True)
            ab_pred = ab_pred - ab_pred_mean

            a_channel = ab_pred[0, :, :, 0]
            b_channel = ab_pred[0, :, :, 1]
            print(f"a_channel shape: {a_channel.shape}, b_channel shape: {b_channel.shape}")

            sepia_mask = (a_channel > 3) & (b_channel > 3)
            a_channel[sepia_mask] = a_channel[sepia_mask] * 0.02
            b_channel[sepia_mask] = b_channel[sepia_mask] * 0.02
            
            a_channel = a_channel - 10
            b_channel = b_channel - 20
            
            for channel in range(2):
                ab_pred[0, :, :, channel] = exposure.equalize_hist(ab_pred[0, :, :, channel])
            
            ab_pred = ab_pred * 2.5
            ab_pred = np.clip(ab_pred, -128, 128)
            ab_pred = torch.tensor(ab_pred).to(device)

            img_l = img_l * 100.0
            result = torch.cat([img_l, ab_pred], dim=1).permute(0, 2, 3, 1).cpu().numpy()[0]
            predicted_rgb = lab2rgb(result)

        predicted_img = Image.fromarray((predicted_rgb * 255).astype(np.uint8))
        predicted_img_resized = predicted_img.resize((orig_width, orig_height), Image.Resampling.LANCZOS)

        enhancer = ImageEnhance.Color(predicted_img_resized)
        predicted_img_enhanced = enhancer.enhance(3.5)
        predicted_path = os.path.join(output_dir, f'{os.path.basename(img_path).split(".")[0]}_predicted.png')
        predicted_img_enhanced.save(predicted_path, format='PNG')
        print(f"Saved predicted image: {predicted_path}")

        gray_img = img.convert('L').convert('RGB')

        collage_width = orig_width
        collage_height = orig_height
        collage = Image.new('RGB', (collage_width * 2, collage_height))

        collage.paste(gray_img, (0, 0))
        collage.paste(predicted_img_enhanced, (collage_width, 0))

        collage_path = os.path.join(output_dir, f'{os.path.basename(img_path).split(".")[0]}_collage.png')
        collage.save(collage_path, format='PNG')
        print(f"Saved collage image: {collage_path}")

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        continue

print("Processing completed!")