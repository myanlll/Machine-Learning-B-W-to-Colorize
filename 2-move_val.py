import os
import random
import shutil
import sys

input_dir = os.path.join('data', 'processed_train')
output_dir = os.path.join('data', 'processed_val')
os.makedirs(output_dir, exist_ok=True)

all_images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
val_size = 10000
val_images = random.sample(all_images, min(len(all_images), val_size))

for idx, image in enumerate(val_images):
    src_path = os.path.join(input_dir, image)
    dst_path = os.path.join(output_dir, image)
    shutil.copyfile(src_path, dst_path)
    progress = (idx + 1) / val_size * 100
    bar_length = 50
    filled_length = int(bar_length * progress // 100)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\rCopying: [{bar}] {progress:.2f}%")
    sys.stdout.flush()

print("\nValidation set created successfully!")