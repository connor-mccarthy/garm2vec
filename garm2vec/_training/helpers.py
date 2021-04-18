import os
import time
from typing import Any, Callable, Dict, Tuple

import pandas as pd
import tensorflow as tf
from garm2vec.constants import (
    DATA_DIR,
    EMBEDDING_DATASET,
    IMAGE_DIR,
    IMAGE_NET_IMAGE_SIZE,
    TOTAL_SAMPLES,
)


def decode_img(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, IMAGE_NET_IMAGE_SIZE)


def process_path(file_id: bytes) -> tf.Tensor:
    img = tf.io.read_file(os.path.join(IMAGE_DIR, f"{bytes.decode(file_id)}.jpg"))
    img = decode_img(img)
    return img


def string_path_to_image(path: tf.Tensor) -> tf.Tensor:
    return tf.numpy_function(process_path, [path], tf.float32)


def dict_dataset_callable_mapper(
    key: str, function: Callable
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def transformation_function(dictionary: Dict[str, Any]) -> Dict[str, Any]:
        return {**dictionary, **{key: function(dictionary[key])}}

    return transformation_function


def get_split_sizes(n_samples: int) -> Tuple[int, int, int]:
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    test_size = int(n_samples * 0.15)
    return train_size, val_size, test_size


def split_dataset(
    dataset: tf.data.Dataset,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_size, val_size, test_size = get_split_sizes(TOTAL_SAMPLES)
    train = dataset.take(train_size)
    test = dataset.skip(train_size)
    val = test.skip(val_size)
    test = test.take(test_size)
    return train, val, test


def read_data() -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, "data.csv")
    with open(filepath, "r") as f:
        file = f.readlines()
    file = [row.strip() for row in file]
    file = [row.split(",", maxsplit=9) for row in file]  # type: ignore

    df = pd.DataFrame(file)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    images = os.listdir(os.path.join(IMAGE_DIR))
    images = [os.path.splitext(image_name)[0] for image_name in images]
    df = df[df["id"].isin(images)]
    df = df.sample(frac=1, random_state=0)
    return df


def benchmark_io(dataset: tf.data.Dataset, num_epochs=2) -> None:
    start_time = time.perf_counter()
    for _ in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)


def configure_dataset_for_performance(
    ds: tf.data.Dataset, batch_size: int
) -> tf.data.Dataset:
    return ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def read_embedding_data() -> pd.DataFrame:
    return pd.read_csv(EMBEDDING_DATASET).iloc[:, 1:].to_numpy()
