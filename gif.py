import os
import io
from PIL import Image
from tbparse import SummaryReader
import imageio
import numpy as np

log_dir = 'runs/GAN_MNIST/fake'
output_dir = 'extracted_images'
os.makedirs(output_dir, exist_ok=True)

reader = SummaryReader(log_dir)
df = reader.images  # pandas DataFrame

print("Available columns:", df.columns)
print(df.head())

extracted_images = []
for idx, row in df.iterrows():
    tag = row['tag']
    step = row['step']
    if tag != 'MNIST fake img':
        continue
    # Try 'image' first, then 'value'
    img_data = row['image'] if 'image' in df.columns else row['value']
    # Try to open as image bytes
    try:
        img = Image.open(io.BytesIO(img_data))
    except Exception:
        # If that fails, try as numpy array
        if isinstance(img_data, np.ndarray):
            arr = img_data
            if arr.dtype != np.uint8:
                arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            print(f"Could not decode image at step {step}")
            continue
    img_path = os.path.join(output_dir, f"{tag.replace(' ', '_')}_step_{step}.png")
    img.save(img_path)
    extracted_images.append(img_path)

# Sort and create GIF
extracted_images.sort()
images = [imageio.imread(img) for img in extracted_images]
imageio.mimsave('gan_progress.gif', images, duration=0.5)

print(f"Extracted {len(extracted_images)} images and saved GIF as gan_progress.gif")
