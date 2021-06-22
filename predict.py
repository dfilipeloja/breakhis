import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_dir = './validation'
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(400, 400),
        batch_size=30,
        class_mode='binary'
    )

model = load_model('./models/breakhis_vgg19_model.h5')

model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy'])

filename = './validation/malignant/SOB_M_DC-14-3909-200-018.png'

img = image.load_img(
        filename,
        target_size=(400, 400))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

pred = model.predict(x[0:])

if pred >= 0.5:
    print(pred, f'{filename} is malignant')
else:
    print(pred, f'{filename} is benign')