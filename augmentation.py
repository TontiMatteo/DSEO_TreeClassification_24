import numpy as np
from scipy.ndimage import rotate
import random

def random_augment_image(image, crop=False):
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

def augment_image(image):
    image_1 = np.flip(image, axis=0)
    image_2 = np.flip(image, axis=1)

    image_3 = np.rot90(image, k=1, axes=(0, 1))
    #image_4 = np.rot90(image, k=2, axes=(0, 1))
    #image_5 = np.rot90(image, k=3, axes=(0, 1))

    # Add small Gaussian noise (helps generalization)
    noise = np.random.normal(0, 0.02, image.shape)
    image_6 = np.clip(image + noise, 0, 1)

    # Random brightness/contrast adjustment
    bright_factor = np.random.uniform(0.8, 1.2)  
    image_bands = image[..., :-3] * bright_factor  # Modify only B bands
    image_bands = np.clip(image_bands, 0, 1)  # Keep in [0,1] range
    image_aug = np.concatenate([image_bands, image[..., -3:]], axis=-1)

    return image_1, image_2, image_3, image_6, image_aug

    