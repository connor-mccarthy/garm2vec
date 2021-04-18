from typing import Dict

import numpy as np
import pytest
import tensorflow as tf
from garm2vec._training.model_components import get_base_model, get_inference_model

from tests.helpers import load_test_image


@pytest.fixture
def inference_model() -> tf.keras.Model:
    return get_inference_model()


@pytest.fixture
def base_model() -> tf.keras.Model:
    return get_base_model()


@pytest.fixture
def learned_embeddings() -> Dict[str, np.ndarray]:
    inference_model = get_inference_model()
    high_heels_args = [load_test_image("high_heels.jpg"), tf.constant(["High heels"])]
    sun_dress_args = [
        load_test_image("sun_dress.jpg"),
        tf.constant(["Summer sun dress"]),
    ]
    ball_dress_args = [
        load_test_image("ball_dress.jpg"),
        tf.constant(["Ball gown"]),
    ]
    winter_hat_args = [load_test_image("winter_hat.jpg"), tf.constant(["Winter hat"])]
    winter_gloves_args = [
        load_test_image("winter_gloves.jpg"),
        tf.constant(["Winter gloves"]),
    ]
    suit_args = [
        load_test_image("suit.jpg"),
        tf.constant(["mens suit"]),
    ]
    tie_args = [
        load_test_image("tie.jpg"),
        tf.constant(["mens tie"]),
    ]

    high_heels_vector = inference_model.predict(high_heels_args)[0]
    sun_dress_vector = inference_model.predict(sun_dress_args)[0]
    ball_dress_vector = inference_model.predict(ball_dress_args)[0]
    winter_gloves_vector = inference_model.predict(winter_gloves_args)[0]
    winter_hat_vector = inference_model.predict(winter_hat_args)[0]
    suit_vector = inference_model.predict(suit_args)[0]
    tie_vector = inference_model.predict(tie_args)[0]

    return {
        "high_heels": high_heels_vector,
        "sun_dress": sun_dress_vector,
        "ball_dress": ball_dress_vector,
        "winter_gloves": winter_gloves_vector,
        "winter_hat": winter_hat_vector,
        "suit": suit_vector,
        "tie": tie_vector,
    }


@pytest.fixture
def pre_learned_embeddings() -> Dict[str, np.ndarray]:
    base_model = get_base_model()
    high_heels_args = [load_test_image("high_heels.jpg"), tf.constant(["High heels"])]
    sun_dress_args = [
        load_test_image("sun_dress.jpg"),
        tf.constant(["Summer sun dress"]),
    ]
    ball_dress_args = [
        load_test_image("ball_dress.jpg"),
        tf.constant(["Ball gown"]),
    ]
    winter_hat_args = [load_test_image("winter_hat.jpg"), tf.constant(["Winter hat"])]
    winter_gloves_args = [
        load_test_image("winter_gloves.jpg"),
        tf.constant(["Winter gloves"]),
    ]
    suit_args = [
        load_test_image("suit.jpg"),
        tf.constant(["mens suit"]),
    ]
    tie_args = [
        load_test_image("tie.jpg"),
        tf.constant(["mens tie"]),
    ]

    high_heels_vector = base_model.predict(high_heels_args)[0]
    sun_dress_vector = base_model.predict(sun_dress_args)[0]
    ball_dress_vector = base_model.predict(ball_dress_args)[0]
    winter_gloves_vector = base_model.predict(winter_gloves_args)[0]
    winter_hat_vector = base_model.predict(winter_hat_args)[0]
    suit_vector = base_model.predict(suit_args)[0]
    tie_vector = base_model.predict(tie_args)[0]

    return {
        "high_heels": high_heels_vector,
        "sun_dress": sun_dress_vector,
        "ball_dress": ball_dress_vector,
        "winter_gloves": winter_gloves_vector,
        "winter_hat": winter_hat_vector,
        "suit": suit_vector,
        "tie": tie_vector,
    }
