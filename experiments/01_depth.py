from pathlib import Path

from PIL import Image
from transformers import pipeline

import numpy as np

# defining the paths
INPUT_PATH = "data/input/cat_01.jpg"
OUTPUT_PATH = "data/output/test_cat_depth_v2.png"

# importing the image
image = Image.open(INPUT_PATH).convert("RGB")

print("Image loaded")
print("Size: ", image.size)

# Loading depth-anything-small model

print("loading depth model...")

depth_estimator = pipeline(
    "depth-estimation",
    model="LiheYoung/depth-anything-small-hf"
)

print("Model loaded")

# Running the model on the image

print("Running depth estimation...")

result = depth_estimator(image)

print("Depth estimation complete")

print("\nResult: ")
print(result.keys())

# Storing and saving the estimation
predicted_depth = result["predicted_depth"]
depth_array = predicted_depth.detach().cpu().numpy()
depth_image = result["depth"]

depth_image.save(OUTPUT_PATH)
print("Saved depth map to: ", OUTPUT_PATH)

np.save("data/output/cat_01_depth.npy", depth_array)
