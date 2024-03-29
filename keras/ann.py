import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from notify import send_message_over_text

from itertools import chain
from keras import datasets, layers, models
import matplotlib.pyplot as plt

seed = 415
batch_size = 32
dropout_value = 0.4

train_images = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    image_size=(48, 48),
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    color_mode='grayscale',
    seed=seed
)

testing_images = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    image_size=(48, 48),
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    color_mode='grayscale',
    seed=seed
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

IMG_SIZE = 48

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])


# build the model
model = models.Sequential([
    # convolutional  starting size 48x48
    tf.keras.layers.Rescaling(1. / 255),

    resize_and_rescale,
    data_augmentation,

    # flat layers
    layers.Flatten(),
    layers.Dense(48*48, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),


    layers.Dense(48*48, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(dropout_value),
    layers.BatchNormalization(),

    layers.Dense(6, activation="softmax")

])


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

history = model.fit(train_images, epochs=100,
                    validation_data=testing_images,
                    callbacks=my_callbacks)

test_loss, test_acc = model.evaluate(testing_images, verbose=2)
print(test_loss, test_acc)

send_message_over_text(f'test loss:  {test_loss} \n{test_acc}')

print(model.summary())