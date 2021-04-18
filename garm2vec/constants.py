import os

MODEL_NAME = "garm2vec_model"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
TRAINED_EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
IMAGE_NET_IMAGE_SIZE = (224, 224)
IMAGE_NET_CHANNELS = 3
EMBEDDING_LAYER_NAME = "garm_embedding"
EMBEDDING_MODEL_NAME = "garm_vector_model"
PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
EMBEDDING_DIMS = 300
TOTAL_SAMPLES = 44_446
BATCH_SIZE = 32
EMBEDDING_DATASET = os.path.join(DATA_DIR, "embeddings.csv")
