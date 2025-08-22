import os
import time
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Required for M1 Mac GPU support

from ultralytics import YOLO
from pathlib import Path

# Load the trained model
model = YOLO('runs/detect/custom_yolo_186_images/weights/best.pt')

# Path to image(s) you want to detect stars in
# Can be a single image or a folder of images
image_path = str(Path(__file__).parent / 'ai_test_image.jpg')
print(image_path)

# Run prediction (inference)
results = model.predict(source=image_path, save=True, conf=0.01, device="mps", show=True, line_width=2)
time.sleep(10)
# `save=True` will save output images with bounding boxes
# `show=True` will open a window to display them (optional)
