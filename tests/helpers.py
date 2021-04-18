import os

import numpy as np
import tensorflow as tf
from garm2vec.helpers import load_image
from scipy import spatial

from tests.constants import TEST_FILES_DIR


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return 1 - spatial.distance.cosine(vector1, vector2)


def load_test_image(image_name: str) -> np.ndarray:
    img = load_image(os.path.join(TEST_FILES_DIR, image_name))
    return tf.expand_dims(img, axis=0)
