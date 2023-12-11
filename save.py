import tensorflow_datasets as tfds
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, losses, regularizers
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Rescaling
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 16
IMAGE_SIZE = 128
SEED = 42
AUTOENCODER_EPOCHS = 3
CLASSIFIER_EPOCHS = 100
num_classes = 5
class_names = ["cardboard", "glass", "metal", "paper", "plastic"]
WEIGHT_DECAY = 0.001
dataset_dir = "./dataset"
train, validate = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=SEED,
    subset="both",
    validation_split=0.2,
    interpolation='bilinear',

)
normalization_layer = tf.keras.layers.Rescaling(1./255)
flip_layer = tf.keras.layers.RandomFlip("horizontal_and_vertical")
rotation_layer = tf.keras.layers.RandomRotation(0.2)


def change_inputs(images, labels):
    x = flip_layer(normalization_layer(images))
    x = rotation_layer(x)

    x = tf.image.resize(x, [IMAGE_SIZE, IMAGE_SIZE],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return x, x


normalized_ds = train.map(change_inputs)

# Autoencoder Model


def create_autoencoder_model():
    input_layer = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # perform downsampling, heading towards bottleneck
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # exiting bottleneck, perform upsampling
    x = tf.keras.layers.Dropout(0.5)(encoded)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', )
    return autoencoder


# Training Autoencoder
autoencoder_model = create_autoencoder_model()
history_autoencoder = autoencoder_model.fit(
    normalized_ds, epochs=AUTOENCODER_EPOCHS)

# Encode data using the trained autoencoder
encoded_data = autoencoder_model.predict(normalized_ds)
print(f"Encoded Data Shape: {encoded_data.shape}")


num_folds = 5
stratified_kfold = StratifiedKFold(
    n_splits=num_folds, shuffle=True, random_state=42)


def create_classifier_model(autoencoder_model):
    # Use the encoder part of the autoencoder as a feature extractor
    encoder_output = autoencoder_model.layers[4].output
    # Add classification layers on top of the encoder output
    x = layers.Flatten()(encoder_output)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.L2(0.001))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu',
                     activity_regularizer=regularizers.L2(0.001))(x)

    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    classifier_model = Model(autoencoder_model.input, output_layer)
    classifier_model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier_model


# Pass your autoencoder model when creating the classifier
classifier_model = create_classifier_model(autoencoder_model)
history_classifier = classifier_model.fit(
    train, epochs=CLASSIFIER_EPOCHS, validation_data=validate)
