import numpy as np
from garm2vec import Garm2Vec
from garm2vec.constants import EMBEDDING_DIMS


def test_garm2vec_runs():
    image_path = "/Users/connor/workspace/garm2vec/data/images/1163.jpg"
    description = "Turtle Check Men Navy Blue Shirt"
    g2v = Garm2Vec()
    vector = g2v.get_vector(image_path, description)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (EMBEDDING_DIMS,)
