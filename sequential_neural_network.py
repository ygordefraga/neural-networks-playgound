from random import randint

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from utils import *


def create_train_set():
    train_labels = []
    train_samples = []

    for i in range(50):
        # The ~5% of younger individuals who did experience side effects
        train_samples.append(randint(13, 64))
        train_labels.append(1)

        # The ~5% of older individuals who did not experience side effects
        train_samples.append(randint(65, 100))
        train_labels.append(0)

    for i in range(1000):
        # The ~95% of younger individuals who did not experience side effects
        train_samples.append(randint(13, 64))
        train_labels.append(0)

        # The ~95% of older individuals who did experience side effects
        train_samples.append(randint(65, 100))
        train_labels.append(1)

    train_samples, train_labels = shuffle(np.array(train_samples), np.array(train_labels))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
    return scaled_train_samples, train_labels


def create_test_set():
    test_labels = []
    test_samples = []

    for i in range(10):
        test_samples.append(randint(13, 64))
        test_labels.append(1)

        test_samples.append(randint(65, 100))
        test_labels.append(0)

    for i in range(200):
        test_samples.append(randint(13, 64))
        test_labels.append(0)

        test_samples.append(randint(65, 100))
        test_labels.append(1)

    test_labels = np.array(test_labels)
    test_samples = np.array(test_samples)
    test_labels, test_samples = shuffle(test_labels, test_samples)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))
    return scaled_test_samples, test_labels


def run_model():
    scaled_train_samples, train_labels = create_train_set()

    model = Sequential([
        Dense(units=16, input_shape=(1,), activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=2, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )

    model.fit(
        x=scaled_train_samples, y=train_labels, validation_split=0.1, shuffle=True, batch_size=10, epochs=30,
        verbose=2
    )

    model.save('models/sequential_neural_network.h5')
    # load_model('models/sequential_neural_network.h5')

    scaled_test_samples, test_labels = create_test_set()

    predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
    rounded_predictions = np.argmax(predictions, axis=-1)

    cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
    plot_cm(cm)


if __name__ == '__main__':
    run_model()
