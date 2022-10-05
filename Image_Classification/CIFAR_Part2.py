from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Activation, BatchNormalization, \
    MaxPooling2D, Flatten, Dropout
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


def load_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test


def data_split():
    X_train, y_train, X_test, y_test = load_dataset()
    (X_train, X_valid) = X_train[5000:], X_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def preprocessing():
    X_train, X_test, X_valid, y_train, y_test, y_valid = data_split()
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_valid = (X_valid - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_valid = to_categorical(y_valid, num_classes=num_classes)
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def augmentation():
    X_train, X_test, X_valid, y_train, y_test, y_valid = preprocessing()
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True,
                                 vertical_flip=False)
    datagen.fit(X_train)
    return datagen, X_train, X_test, X_valid, y_train, y_test, y_valid


def model_creation():
    datagen, X_train, X_test, X_valid, y_train, y_test, y_valid = augmentation()
    base_hidden_layer = 32
    weight_decay = 1e-4
    model = Sequential()

    # CONV-1
    model.add(Conv2D(filters=base_hidden_layer, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # CONV-2
    model.add(Conv2D(filters=base_hidden_layer, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # POOL + Dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # CONV-3
    model.add(Conv2D(filters=base_hidden_layer * 2, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # CONV-4
    model.add(Conv2D(filters=base_hidden_layer * 2, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # POOL + Dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # CONV-5
    model.add(Conv2D(filters=base_hidden_layer * 4, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # CONV-6
    model.add(Conv2D(filters=base_hidden_layer * 4, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # POOL + Dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # FC-7
    model.add(Flatten())
    model.add(Dense(10., activation='softmax'))

    return model, datagen, X_train, X_test, X_valid, y_train, y_test, y_valid


def train_model():
    model, datagen, X_train, X_test, X_valid, y_train, y_test, y_valid = model_creation()
    batch_size = 128
    epochs = 10
    checkpointer = ModelCheckpoint(filepath='model.CIFAR_PART2.hdf5', verbose=2,
                                   save_best_only=True)
    optimizer = optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        callbacks=[checkpointer],
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=epochs, verbose=2, validation_data=(X_valid, y_valid))
    return history, model, X_train, X_test, X_valid, y_train, y_test, y_valid


def predict():

    history, model, X_train, X_test, X_valid, y_train, y_test, y_valid = train_model()

    score = model.evaluate(X_test, y_test, verbose=1, batch_size=128)
    print(score[1] * 100)


predict()
