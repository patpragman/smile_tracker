import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from itertools import chain
from keras import datasets, layers, models
import matplotlib.pyplot as plt

seed = 415
batch_size = 64

train_images = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    image_size=(48, 48),
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=seed
)

testing_images = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    image_size=(48, 48),
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=seed
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

filter_shape = (4, 4)
filter_count = 32
dropout_value = 0.18
pooling_value = 4


# build the model
model = models.Sequential([
    # convolutional  starting size 48x48
    tf.keras.layers.Rescaling(1. / 255),  # size 45x45 inexplicably

    layers.Conv2D(64, (4, 4), activation='relu', input_shape=(48, 48, 3), strides=1),
    layers.Dropout(dropout_value),
    layers.MaxPooling2D(2, strides=1),

    layers.Conv2D(64, (2, 2), activation='relu', strides=1),
    layers.Dropout(dropout_value),

    layers.Conv2D(64, (2, 2), activation='relu', strides=2),
    layers.Dropout(dropout_value),

    layers.Conv2D(64, (2, 2), activation='relu', strides=1),
    layers.Dropout(dropout_value),
    layers.MaxPooling2D(2, strides=2),

    layers.Conv2D(64, (2, 2), activation='relu', strides=1),
    layers.Dropout(dropout_value),
    layers.MaxPooling2D(2, strides=2),

    # flat layers
    layers.Flatten(),
    layers.Dense(100, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(100, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(100, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(100, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(100, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(100, activation="relu"),
    layers.Dropout(dropout_value),


    layers.Dense(25, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(7, activation="relu"),

    layers.Dense(7, activation="softmax")

])


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

history = model.fit(train_images, epochs=100,
                    validation_data=testing_images,
                    callbacks=my_callbacks)

print(model.summary())
test_loss, test_acc = model.evaluate(testing_images, verbose=2)
print(test_loss, test_acc)