import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from keras import datasets, layers, models
import matplotlib.pyplot as plt

seed = 415

train_images_loader = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    labels="inferred",
    image_size=(48, 48),
    batch_size=256,
    subset="training",
    validation_split=0.2,
    seed=seed
)

testing_images_loader = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    labels="inferred",
    image_size=(48, 48),
    batch_size=64,
    subset="validation",
    validation_split=0.2,
    seed=seed
)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]



# build the model
model = keras.Sequential([
    layers.Rescaling(1./255),
    keras.layers.Dense(2304, activation='relu'),
    layers.Dropout(.2),
    keras.layers.Dense(2304, activation='relu'),
    layers.Dropout(.2),
    keras.layers.Dense(100, activation='relu'),
    layers.Dropout(.2),
    keras.layers.Dense(7, activation='softmax')
])



model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy']
              )

history = model.fit(train_images_loader, epochs=100,
                    validation_data=testing_images_loader,
                    callbacks=my_callbacks)

test_loss, test_acc = model.evaluate(testing_images_loader, verbose=2)
