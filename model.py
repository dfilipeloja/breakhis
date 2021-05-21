from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense


def sequential():
    classifier = Sequential()

    classifier.add(Convolution2D(32, (3, 3), input_shape=(350, 230, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units= 600, activation='relu'))

    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier
