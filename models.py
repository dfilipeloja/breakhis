from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow import optimizers


class Models:

    @staticmethod
    def alexnet(image_height, image_width, channels):
        model = Sequential()
        # 1st Convolutional Layer
        model.add(
            Conv2D(filters=96, input_shape=(image_height, image_width, channels), kernel_size=(11, 11), strides=(4, 4),
                   padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(32, 32, 3,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(1))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def vgg19_pretrained(image_height, image_width, channels):
        vggmodel = VGG19(include_top=False, weights='imagenet', input_shape=(image_height, image_width, channels))
        vggmodel.trainable = True

        for layer in vggmodel.layers:
            layer.trainable = True if layer.name == 'block5_conv1' else False

        model = Sequential()

        model.add(vggmodel)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))

        model.add(Dense(1, activation='sigmoid', name='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=1e-5),
                      metrics=['accuracy'])

        return model
