import tensorflow as tf


def get_otrain(input_shape):
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding='same', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding='same'),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return cnn
