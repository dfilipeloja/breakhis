from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_dir = './validation'
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(400, 400),
        batch_size=30,
        class_mode='binary'
    )

model = load_model('./models/breakhis_vgg19_model.h5')

Y_pred = model.predict(test_generator)
