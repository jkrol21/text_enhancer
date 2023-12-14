from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageEnhance


def increase_image_size(img, iterations=1):
    out_size = (img.size[0] * 2**iterations, img.size[1] * 2**iterations)
    upscaled_image = img.resize(size=out_size, resample=PIL.Image.LANCZOS)

    return upscaled_image


def preprocess_image(image: PIL.Image) -> np.ndarray:
    image_array = np.array(image)
    image_array = image_array[:, :, 1]
    image_array = image_array / 255.0
    image_array = 1 - image_array
    image_array = np.expand_dims(image_array, axis=-1)

    return image_array


def image_from_prediction(image_array: np.ndarray) -> PIL.Image:
    image_array = np.squeeze(image_array, axis=-1)
    image_array = ((1 - image_array) * 255).astype(np.uint8)
    image_array = np.dstack([image_array] * 3)
    image = Image.fromarray(image_array)

    return image


def generate_prediction(model, input: np.ndarray) -> PIL.Image:
    prediction = model.predict(input)
    prediction = image_from_prediction(prediction)

    return predictions


def enhance_image(image: PIL.Image, model, increase_size=True) -> PIL.Image:
    if increase_size:
        image = increase_size_image(image)

    X_image = np.array([preprocess_image(image)])
    enhanced_image = generate_prediction(model, X_image)

    return enhanced_image
