from models import Models
from graphs import Graph
from keras.preprocessing.image import ImageDataGenerator
import os

train_dir = './train'
test_dir = './test'
results_dir = './results'
models_dir = './models'

image_height, image_width = 400, 400
batch_size = 30
channels = 3

alexnet_model = Models.alexnet(image_height, image_width, channels)
alexnet_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

epochs = 50
num_train = 1608
num_test = 405 # (125 benign and 280 malignant)

model_fit = alexnet_model.fit(
    train_generator,
    steps_per_epoch=num_train//batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=num_test//batch_size
)

alexnet_model.save(os.path.join(models_dir, 'breakhis_alexnet_model.h5'))
Graph.plot_accuracy_loss(model_fit, results_dir)