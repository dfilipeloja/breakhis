import os

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

model = load_model('./models/breakhis_vgg19_model.h5')

model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy'])

#filename = './validation/malignant/SOB_M_DC-14-3909-200-018.png'
test_dir = './test'

test_datagen = ImageDataGenerator()
gen = test_datagen.flow_from_directory(
    test_dir
)

images = []
counter = 0

for filename in gen.filenames:
    img = image.load_img(
            filename,
            target_size=(400, 400))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    images.append(img)

print(images)
# img = image.load_img(
#         filename,
#         target_size=(400, 400))
#
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = x / 255.0
#
# print(type(x))
# print(x.shape)
#
# pred = model.predict(x)
#
# if pred >= 0.5:
#     print(pred, f'{filename} is malignant')
# else:
#     print(pred, f'{filename} is benign')