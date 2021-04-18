from typing import Any, Dict, Iterable, Iterator, Tuple

import tensorflow as tf
from garm2vec._training.config import classifiers
from garm2vec._training.helpers import (
    benchmark_io,
    dict_dataset_callable_mapper,
    read_data,
    read_embedding_data,
    split_dataset,
    string_path_to_image,
)
from garm2vec.constants import BATCH_SIZE
from sklearn import preprocessing


def create_input_dataset() -> tf.data.Dataset:
    df = read_data()
    texts = tf.convert_to_tensor(df["productDisplayName"].to_numpy())
    image_id_generator = (i for i in df["id"].to_numpy())

    def input_generator() -> Iterator[Dict[str, Any]]:
        for i, t in zip(image_id_generator, texts):
            yield {"image": tf.constant(i), "text": tf.constant(t)}

    input_signature = {
        "image": tf.TensorSpec(shape=(), dtype=tf.string),
        "text": tf.TensorSpec(shape=(), dtype=tf.string),
    }
    return tf.data.Dataset.from_generator(
        input_generator, output_signature=input_signature
    ).map(
        dict_dataset_callable_mapper("image", string_path_to_image),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def create_vectorized_input_dataset() -> tf.data.Dataset:
    embeddings = read_embedding_data()

    def vectorized_input_generator() -> Iterator[Dict[str, Any]]:
        for embedding in embeddings:
            yield {"input_embedding": tf.constant(embedding)}

    input_signature = {
        "input_embedding": tf.TensorSpec(shape=(2560,), dtype=tf.float32),
    }
    return tf.data.Dataset.from_generator(
        vectorized_input_generator, output_signature=input_signature
    )


def create_output_dataset() -> tf.data.Dataset:
    df = read_data()
    lb = preprocessing.LabelBinarizer()
    gender = lb.fit_transform(df["gender"])
    master_category = lb.fit_transform(df["masterCategory"])
    article_type = lb.fit_transform(df["articleType"])
    season = lb.fit_transform(df["season"])
    usage = lb.fit_transform(df["usage"])

    def output_generator() -> Iterator[Dict[str, Any]]:
        for s, g, at, u, mc in zip(
            season,
            gender,
            article_type,
            usage,
            master_category,
        ):
            yield {
                "season": tf.convert_to_tensor(s),
                "gender": tf.convert_to_tensor(g),
                "article_type": tf.convert_to_tensor(at),
                "usage": tf.convert_to_tensor(u),
                "master_category": tf.convert_to_tensor(mc),
            }

    output_signature = {
        classifier.name: tf.TensorSpec(shape=(classifier.nodes,), dtype=tf.int64)
        for classifier in classifiers
    }

    return tf.data.Dataset.from_generator(
        output_generator, output_signature=output_signature
    )


def create_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    input_dataset = create_vectorized_input_dataset()
    output_dataset = create_output_dataset()
    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    train, val, test = split_dataset(dataset)
    return train, val, test


def create_dataset_from_strings(image_paths: Iterable[str], texts: Iterable[str]):
    image_generator = (i for i in image_paths)
    text_generator = (i for i in texts)

    def input_generator() -> Iterator[Dict[str, Any]]:
        for i, t in zip(image_generator, text_generator):
            yield {"image": tf.constant(i), "text": tf.constant(t)}

    input_signature = {
        "image": tf.TensorSpec(shape=(), dtype=tf.string),
        "text": tf.TensorSpec(shape=(), dtype=tf.string),
    }
    return tf.data.Dataset.from_generator(
        input_generator, output_signature=input_signature
    ).map(
        dict_dataset_callable_mapper("image", string_path_to_image),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


if __name__ == "__main__":
    train, val, test = create_datasets()
    # train = tf.data.Dataset.range(2).interleave(
    #     lambda _: train, num_parallel_calls=tf.data.AUTOTUNE
    # ) # another option for performance tuning dataset via parallelization
    train = train.take(640)
    train = train.batch(BATCH_SIZE)
    benchmark_io(train)

    train = train.batch(2).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    benchmark_io(train)

    for i in train.take(1):
        print(i)
