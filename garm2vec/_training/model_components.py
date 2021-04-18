import os
from typing import List, Tuple

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401
from garm2vec._training.config import classifiers
from garm2vec.constants import (
    EMBEDDING_LAYER_NAME,
    EMBEDDING_MODEL_NAME,
    IMAGE_NET_CHANNELS,
    IMAGE_NET_IMAGE_SIZE,
    PROJECT_ROOT,
    TRAINED_EMBEDDING_MODEL_PATH,
)
from tensorflow.keras.applications import resnet50

SMALL_BERT = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
PREPROCESSING_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


def build_image_model() -> tf.keras.Model:
    preprocess_input = resnet50.preprocess_input
    resnet_model = resnet50.ResNet50(weights="imagenet", include_top=False)
    global_max_pooling = tf.keras.layers.GlobalAveragePooling2D()

    image_inputs = tf.keras.Input(
        shape=IMAGE_NET_IMAGE_SIZE + (IMAGE_NET_CHANNELS,), name="image"
    )
    x = preprocess_input(image_inputs)
    x = resnet_model(x, training=False)
    image_embedding = global_max_pooling(x)
    return tf.keras.Model(image_inputs, image_embedding, name="ResNet50")


def get_image_input_output() -> Tuple[tf.Tensor, tf.Tensor]:
    image_model = build_image_model()
    return image_model.inputs, image_model.output


def build_language_model() -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(PREPROCESSING_MODEL, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(SMALL_BERT)
    encoder.trainable = False
    outputs = encoder(encoder_inputs)
    document_embedding = outputs["pooled_output"]
    return tf.keras.Model(text_input, document_embedding, name="Small_BERT")


def get_language_input_output() -> Tuple[tf.Tensor, tf.Tensor]:
    language_model = build_language_model()
    return language_model.inputs, language_model.output


def get_base_model() -> tf.keras.Model:
    image_input, image_output = get_image_input_output()
    language_input, language_output = get_language_input_output()
    concatenated_output = tf.keras.layers.concatenate([image_output, language_output])
    return tf.keras.Model(
        inputs=[image_input, language_input],
        outputs=concatenated_output,
        name="base_model",
    )


def get_embedding_model(n_dims: int = 300) -> tf.keras.Model:
    base_model = get_base_model()
    input_ = tf.keras.layers.Input(
        base_model.output.shape[-1], dtype=base_model.output.dtype
    )
    garm_vector_output = tf.keras.layers.Dense(n_dims, name=EMBEDDING_LAYER_NAME)(
        input_
    )
    return tf.keras.Model(
        inputs=input_,
        outputs=garm_vector_output,
        name=EMBEDDING_MODEL_NAME,
    )


def get_classifier_layers() -> List[tf.keras.layers.Dense]:
    return [
        tf.keras.layers.Dense(
            classifier.nodes, activation=classifier.activation, name=classifier.name
        )
        for classifier in classifiers
    ]


def get_classification_model() -> tf.keras.Model:
    embedding_model = get_embedding_model()
    input_layer = tf.keras.layers.Input(
        embedding_model.input.shape[-1],
        dtype=embedding_model.input.dtype,
        name="input_embedding",
    )
    x = embedding_model(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    classifier_layers = get_classifier_layers()
    outputs = [classifier_layer(x) for classifier_layer in classifier_layers]
    return tf.keras.Model(inputs=input_layer, outputs=outputs, name="classifier_model")


def get_inference_model() -> tf.keras.Model:
    base_model = get_base_model()
    trained_embedding_model = tf.keras.models.load_model(
        TRAINED_EMBEDDING_MODEL_PATH, compile=False
    )

    inputs = base_model.inputs
    x = base_model(inputs)
    output = trained_embedding_model(x)
    return tf.keras.models.Model(
        inputs=inputs,
        outputs=output,
    )


if __name__ == "__main__":
    garm2vec_model = get_inference_model()
    garm2vec_model_path = os.path.join(PROJECT_ROOT, "embedding_architecture.png")
    plot = tf.keras.utils.plot_model(
        garm2vec_model, garm2vec_model_path, expand_nested=True, show_shapes=True
    )

    full_model = get_classification_model()
    full_model_path = os.path.join(PROJECT_ROOT, "full_classification_architecture.png")
    plot = tf.keras.utils.plot_model(
        full_model,
        full_model_path,
        expand_nested=True,
        show_shapes=True,
    )
