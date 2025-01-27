import numpy as np
from scipy.ndimage import rotate
import random

def augment_image(image, crop=False):
    """Apply a series of augmentations to the image."""
    # Example augmentations
    def random_flip(img):
        if random.random() > 0.5:
            img = np.flip(img, axis=1)  # Horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=2)  # Vertical flip
        return img

    def random_rotate(img):
        angle = random.choice([0, 90, 180, 270])
        return rotate(img, angle, axes=(1, 2), reshape=False)

    def add_noise(img, mean=0, std=0.05):
        noise = np.random.normal(mean, std, img.shape)
        return img + noise

    def random_crop_and_resize(image, crop_size=9, target_size=11):
        """Randomly crop and resize the image."""
        c, h, w = image.shape
        start_h = random.randint(0, h - crop_size)
        start_w = random.randint(0, w - crop_size)
        cropped = image[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
        return np.resize(cropped, (c, target_size, target_size))

    # Apply augmentations
    image = random_flip(image)
    image = random_rotate(image)
    image = add_noise(image)
    if crop:
        image = random_crop_and_resize(image)
    return image