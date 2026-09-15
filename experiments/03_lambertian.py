from pathlib import Path
from PIL import Image

import numpy as np

normals = np.load("data/output/cat_01_normal.npy")

light = np.array([1.5, 0.0, 0.0], dtype=np.float32) # light position
light_color = np.array([1.0, 0.15, 0.7], dtype=np.float32) # light color


h, w, _ = normals.shape
y, x = np.mgrid[0:h, 0:w]

x = (x / (w - 1)) * 2.0 - 1.0
y = (y / (h - 1)) * 2.0 - 1.0

depth = np.load("data/output/cat_01_depth.npy")
z = depth

z_min = z.min()
z_max = z.max()

z = (depth - z_min) / (z_max - z_min)
z = z * 2.0 - 1.0

lx = light[0] - x
ly = light[1] - y
lz = light[2] - z

nx = normals[:, :, 0]
ny = normals[:, :, 1]
nz = normals[:, :, 2]

light_magnitude = np.sqrt(
    lx**2 +
    ly**2 +
    lz**2
)

lx /= light_magnitude
ly /= light_magnitude
lz /= light_magnitude

normal_magnitude = np.sqrt(
    nx**2 +
    ny**2 +
    nz**2
)

nx /= normal_magnitude
ny /= normal_magnitude
nz /= normal_magnitude

brightness = (
    nx * lx +
    ny * ly +
    nz * lz
)

brightness = np.maximum(brightness, 0.0)

ambient = 0.2
diffuse = brightness

ambient_color = np.array(
    [1.0, 1.0, 1.0],
    dtype=np.float32
)


view = np.array([0.0, 0.0, 1.0], dtype=np.float32)

print("\nView Direction: ")
print(view)

normal_light_dot = (
    nx * lx +
    ny * ly +
    nz * lz
)

rx = 2.0 * normal_light_dot * nx - lx
ry = 2.0 * normal_light_dot * ny - ly
rz = 2.0 * normal_light_dot * nz - lz

reflection_magnitude = np.sqrt(
    rx**2 +
    ry**2 +
    rz**2
)


reflection_view_dot = (
    rx * view[0] +
    ry * view[1] +
    rz * view[2]
)

reflection_view_dot = np.maximum(
    reflection_view_dot,
    0.0
)

shininess = 20


specular = reflection_view_dot ** shininess

ambient_strength = 0.2
diffuse_strength = 0.7
specular_strength = 0.3

lighting = (
    ambient_strength
    + diffuse_strength * diffuse
    + specular_strength * specular
)

direct = (
    diffuse_strength * brightness[..., None] * light_color
    + specular_strength * specular[..., None] * light_color
)

lighting = (
    ambient_strength + direct
)

illumination = (
    ambient * ambient_color
    + (1.0 - ambient) * direct
)

specular_image = (
    specular * 255
).clip(0, 255).astype(np.uint8)

Image.fromarray(specular_image).save("data/output/test_cat_specular.png")

print("\nBrightness percentiles:")
print(np.percentile(
    brightness,
    [0, 1, 10, 25, 50, 75, 90, 99, 100]
))


brightness_image = (
    brightness * 255
).clip(0, 255).astype(np.uint8)
Image.fromarray(brightness_image).save("data/output/test_cat_brightness.png")

original = np.array(
    Image.open("data/input/person_test_01.jpg").convert("RGB"),
    dtype=np.float32
) / 255.0

relit = original * lighting

relit_image = (
    relit * 255
).clip(0, 255).astype(np.uint8)
Image.fromarray(relit_image).save("data/output/test_cat_relit.png")

print("\nSaved relit image")