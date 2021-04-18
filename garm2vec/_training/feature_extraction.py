import pandas as pd
from garm2vec._training.data import create_input_dataset
from garm2vec._training.helpers import configure_dataset_for_performance, read_data
from garm2vec._training.model_components import get_base_model
from garm2vec.constants import BATCH_SIZE, EMBEDDING_DATASET


def extract_features() -> None:
    df = read_data()
    input_dataset = create_input_dataset()
    input_dataset = configure_dataset_for_performance(input_dataset, BATCH_SIZE)
    base_model = get_base_model()
    predictions = base_model.predict(input_dataset)
    df = pd.DataFrame(predictions)
    df.to_csv(EMBEDDING_DATASET)
    print("Wrote embeddings to:", EMBEDDING_DATASET)


if __name__ == "__main__":
    extract_features()
