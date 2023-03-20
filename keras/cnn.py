import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from notify import send_message_over_text

from itertools import chain
from keras import datasets, layers, models

# remove the disgusted images, because honestly it's not enough data


seed = 415
batch_size = 32

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

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.3),
])

IMG_SIZE = 48

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                         patience=15, verbose=1)
]


dropout_value = 0.25


# build the model
model = models.Sequential([
    # convolutional  starting size 48x48
    resize_and_rescale,
    data_augmentation,  # twisting and scaling

    # first layer, look for features
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), strides=2),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # second layer of feature extraction
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # third conv layer
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 4th layer
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 5th layer
    layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 6th layer
    layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 7th layer
    layers.Conv2D(2048, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 8th layer
    layers.Conv2D(1024, kernel_size=(4, 4), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 9th layer
    layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 10th layer
    layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 11th layer
    layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 12th layer
    layers.Conv2D(512, kernel_size=(4, 4), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 13th layer
    layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 14th layer
    layers.Conv2D(512, kernel_size=(4, 4), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 15th layer
    layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 16th layer
    layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 17th layer
    layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # 18th layer
    layers.Conv2D(128, kernel_size=(2, 2), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    layers.BatchNormalization(),
    layers.Dropout(dropout_value),

    # flat layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(dropout_value),

    layers.Dense(128, activation="relu"),
    layers.Dropout(dropout_value),

    layers.Dense(3, activation="softmax")

])


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

history = model.fit(train_images, epochs=250,
                    validation_data=testing_images,
                    callbacks=my_callbacks)

test_loss, test_acc = model.evaluate(testing_images, verbose=2)
print(test_loss, test_acc)

send_message_over_text(f'test loss:  {test_loss} \n{test_acc}')

print(model.summary())
