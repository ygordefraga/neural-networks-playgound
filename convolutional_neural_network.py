import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import warnings
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    train_path = 'data/dogs-vs-cats/train'
    valid_path = 'data/dogs-vs-cats/valid'
    test_path = 'data/dogs-vs-cats/test'

    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10,
                             shuffle=False)

    imgs, labels = next(train_batches)
    # plot_images(imgs)
    print(labels)

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epochs=10,
        verbose=2
    )

    predictions = model.predict(x=test_batches, verbose=0)
    cm = confusion_matrix(test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plot_cm(cm)
