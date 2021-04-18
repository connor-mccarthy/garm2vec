import numpy as np
from garm2vec import Garm2Vec
from garm2vec.constants import EMBEDDING_DIMS


def test_garm2vec_runs():
    image_path = "/Users/connor/workspace/garm2vec/data/images/1163.jpg"
    description = "Turtle Check Men Navy Blue Shirt"
    inputs = [image_path, description]
    g2v = Garm2Vec()
    vector = g2v.get_one(inputs)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (EMBEDDING_DIMS,)

    vectors = g2v.get_many([inputs, inputs])
    assert isinstance(vectors, np.ndarray)
    assert vector.shape[-1] == EMBEDDING_DIMS

    g2v.plot_one(inputs)
    g2v.plot_many([inputs, inputs])
