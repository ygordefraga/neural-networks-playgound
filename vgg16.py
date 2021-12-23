import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from utils import *


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

    vgg16_model = tf.keras.applications.vgg16.VGG16()

    model = Sequential()

    for layer in vgg16_model.layers[:-1]:
        # The model already knows the weights
        layer.trainable = False
        model.add(layer)

    model.add(Dense(units=2, activation='softmax'))
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=train_batches, steps_per_epoch=len(train_batches), validation_data=valid_batches,
              validation_steps=len(valid_batches), epochs=5, verbose=2)

    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

    cm = confusion_matrix(test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plot_cm(cm)
