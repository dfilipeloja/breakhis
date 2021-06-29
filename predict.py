import os

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

IMG_W = 400
IMG_H = 400

model = load_model('./models/breakhis_vgg19_model.h5')

model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy'])

#filename = './validation/malignant/SOB_M_DC-14-3909-200-018.png'
test_dir = './test'

test_datagen = ImageDataGenerator()
gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_W, IMG_W),
    batch_size=30,
    class_mode='binary'
)

images = []
counter = 0

for img in gen:
    idx = (gen.batch_index - 1) * gen.batch_size
    #print(gen.filenames[idx: idx + gen.batch_size])
    counter += 1

print(counter)
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