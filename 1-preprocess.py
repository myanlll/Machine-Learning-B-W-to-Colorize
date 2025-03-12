from PIL import Image
import os
import time
import sys

def crop_and_resize_to_512(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size
        short_side = min(width, height)
        long_side = max(width, height)
        
        if width < height:
            crop_start = (height - width) // 2
            cropped_img = img.crop((0, crop_start, width, crop_start + width))
        else:
            crop_start = (width - height) // 2
            cropped_img = img.crop((crop_start, 0, crop_start + height, height))
        
        resized_img = cropped_img.resize((512, 512), Image.BICUBIC)
        resized_img.save(output_path)

input_dir = os.path.join('data', 'train_2017')
output_dir = os.path.join('data', 'processed_train')
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
total_files = len(files)

for index, filename in enumerate(files):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    crop_and_resize_to_512(input_path, output_path)
    progress = (index + 1) / total_files * 100
    bar_length = 50
    filled_length = int(bar_length * progress // 100)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\rProcessing: [{bar}] {progress:.2f}%")
    sys.stdout.flush()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nCompleted! Elapsed time: {elapsed_time:.2f} seconds")