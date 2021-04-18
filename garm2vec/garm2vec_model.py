import os
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_text  # noqa: F401

from garm2vec._training.model_components import get_inference_model
from garm2vec.helpers import load_image


class Garm2Vec:
    def __init__(self) -> None:
        self.model = get_inference_model()

    def get_one(self, inputs: Iterable[str]) -> np.array:
        input_arr = np.array(inputs)
        input_arr_2d = np.expand_dims(input_arr, axis=0)
        return self.get_many(input_arr_2d)[0]

    def get_many(self, inputs: Iterable[Iterable[str]]) -> np.ndarray:
        input_arr = np.array(inputs)
        image_paths = input_arr[:, 0]
        images = (np.expand_dims(load_image(path), axis=0) for path in image_paths)
        descriptions_raw = input_arr[:, 1]
        descriptions = (tf.constant([description]) for description in descriptions_raw)
        input_generator = (
            [image, description] for image, description in zip(images, descriptions)
        )
        return self.model.predict(input_generator)

    def plot_many(self, inputs: Iterable[Iterable[str]]):
        input_arr = np.array(inputs)
        vectors = self.get_many(input_arr)
        x = list(range(vectors.shape[-1]))
        fig = go.Figure()
        for i, vector in enumerate(vectors):
            fig.add_trace(go.Bar(x=x, y=list(vector), name=f"Garment {i+1}"))
        fig.show()
        return fig

    def plot_one(self, inputs: Iterable[Iterable[str]]):
        input_arr = np.array(inputs)
        input_arr_2d = np.expand_dims(input_arr, axis=0)
        return self.plot_many(input_arr_2d)


if __name__ == "__main__":
    from garm2vec.constants import IMAGE_DIR

    sample_image_path = os.path.join(IMAGE_DIR, "1163.jpg")
    description = "Turtle Check Men Navy Blue Shirt"
    vector = Garm2Vec().get_one([sample_image_path, description])
    print(vector)
