"""Small image transforms for bright handwritten digits on a dark background."""

import random

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, grey_dilation, grey_erosion, map_coordinates


def geometric_augmentation(image, settings):
    angle = np.deg2rad(random.uniform(-settings.get("rotation", 10), settings.get("rotation", 10)))
    shear = np.deg2rad(random.uniform(-settings.get("shear", 0), settings.get("shear", 0)))
    scale = random.uniform(settings.get("scale_min", 0.85), settings.get("scale_max", 1.15))
    distance = settings.get("translation", 2)
    translation = np.array([random.uniform(-distance, distance), random.uniform(-distance, distance)])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    skew = np.array([[1, np.tan(shear)], [0, 1]])
    inverse = np.linalg.inv(scale * rotation @ skew)
    center = (np.array(image.shape) - 1) / 2
    offset = center - inverse @ (center + translation)
    # Combine all affine operations into one interpolation to limit blurring.
    return affine_transform(image, inverse, offset=offset, order=1,
                            mode="constant", cval=0, prefilter=False)


def elastic_deformation(image, strength=0.7, sigma=4):
    dx = gaussian_filter(np.random.uniform(-1, 1, image.shape), sigma, mode="reflect")
    dy = gaussian_filter(np.random.uniform(-1, 1, image.shape), sigma, mode="reflect")
    # Strength is the RMS length of the displacement vector, in pixels.
    rms = np.sqrt(np.mean(dx * dx + dy * dy))
    dx = dx * strength / (rms + 1e-8)
    dy = dy * strength / (rms + 1e-8)
    rows, columns = np.indices(image.shape)
    return map_coordinates(image, [rows + dy, columns + dx], order=1,
                           mode="constant", cval=0, prefilter=False)


def morphology(image, operation):
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if operation == "erosion":
        return grey_erosion(image, footprint=footprint, mode="constant", cval=0)
    if operation == "dilation":
        return grey_dilation(image, footprint=footprint, mode="constant", cval=0)
    raise ValueError("operation must be erosion or dilation")


def additional_augmentation(image, settings):
    # Extra transforms use NumPy's RNG; geometric transforms use Python's RNG.
    # This preserves the same geometric draws and minibatch order in the ablation.
    probability = settings.get("elastic_probability", 0)
    if probability > 0 and np.random.random() < probability:
        image = elastic_deformation(image, settings.get("elastic_strength", 0.7),
                                     settings.get("elastic_sigma", 4))
    probability = settings.get("morphology_probability", 0)
    if probability > 0 and np.random.random() < probability:
        image = morphology(image, np.random.choice(["erosion", "dilation"]))
    probability = settings.get("blur_probability", 0)
    if probability > 0 and np.random.random() < probability:
        sigma = np.random.uniform(settings.get("blur_sigma_min", 0.2), settings.get("blur_sigma_max", 0.7))
        image = gaussian_filter(image, sigma, mode="constant", cval=0)
    probability = settings.get("photometric_probability", 0)
    if probability > 0 and np.random.random() < probability:
        amount = settings.get("photometric_range", 0.15)
        brightness = np.random.uniform(1 - amount, 1 + amount)
        contrast = np.random.uniform(1 - amount, 1 + amount)
        mean = image.mean()
        image = brightness * ((image - mean) * contrast + mean)
    return np.clip(image, 0, 1).astype(np.float32)
