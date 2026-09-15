from pathlib import Path
from scipy.ndimage import sobel
from PIL import Image

import numpy as np

DEPTH_PATH = Path("data/output/person_test_01.npy")

depth = np.load(DEPTH_PATH)
depth_min = depth.min()
depth_max = depth.max()

depth_norm = (
    (depth - depth_min) / 
    (depth_min - depth_max)
) * 2.0 - 1.0

dz_dx = sobel(depth_norm, axis=1)
dz_dy = sobel(depth_norm, axis=0)

h, w = depth.shape
cy = h // 2
cx = w // 2

nx = -dz_dx
ny = -dz_dy
nz = np.ones_like(depth)

magnitude = np.sqrt(nx**2 + ny**2 + nz**2)

nx /= magnitude
ny /= magnitude
nz /= magnitude

normals = np.stack([nx, ny, nz], axis=-1)
np.save("data/output/person_test_normal.npy", normals)

normal_image = ((normals + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(normal_image).save("data/output/person_test_normal.png")

print("\n Saved Normal Map")
