from datetime import datetime

import tensorflow as tf
import tensorflow_text  # noqa: F401
from garm2vec._training.config import classifiers
from garm2vec._training.data import create_datasets
from garm2vec._training.helpers import configure_dataset_for_performance
from garm2vec._training.model_components import get_classification_model
from garm2vec._training.tensorboard_bug_fix import TBCallback
from garm2vec.constants import (
    BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    TRAINED_EMBEDDING_MODEL_PATH,
)
from livelossplot import PlotLossesKeras
from plot_keras_history import plot_history
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    loss = {classifier.name: classifier.loss for classifier in classifiers}
    metrics = {classifier.name: classifier.metrics for classifier in classifiers}
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


def train_classifier() -> tf.keras.Model:
    model = get_classification_model()
    model = compile_model(model)

    train, val, test = create_datasets()
    train = configure_dataset_for_performance(train, BATCH_SIZE)
    val = configure_dataset_for_performance(val, BATCH_SIZE)
    test = configure_dataset_for_performance(test, BATCH_SIZE)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        TerminateOnNaN(),
        TBCallback(log_dir=log_dir, histogram_freq=1),
        PlotLossesKeras(),
    ]
    fit_kwargs = dict(
        epochs=20,
        shuffle=True,
        callbacks=callbacks,
    )
    history = model.fit(train, validation_data=val, **fit_kwargs)

    plot_history(history.history)

    scores = model.evaluate(test)
    metric_dict = dict(zip(model.metrics_names, scores))
    print("Test metrics:", metric_dict)

    return model


def save_embedding_layer_weights(model: tf.keras.Model) -> None:
    embedding_layer = model.get_layer(EMBEDDING_MODEL_NAME)
    embedding_layer.save(TRAINED_EMBEDDING_MODEL_PATH)
    print("Saved model to", TRAINED_EMBEDDING_MODEL_PATH)


if __name__ == "__main__":
    model = train_classifier()
    save_embedding_layer_weights(model)
