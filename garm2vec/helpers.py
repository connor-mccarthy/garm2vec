import numpy as np
from tensorflow.keras.preprocessing import image

from garm2vec.constants import IMAGE_NET_IMAGE_SIZE


def load_image(image_path: str) -> np.ndarray:
    img = image.load_img(image_path, target_size=IMAGE_NET_IMAGE_SIZE)
    img = image.img_to_array(img)
    return img
