import os

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

IMG_W, IMG_H = 400

model = load_model('./models/breakhis_vgg19_model.h5')

model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy'])

#filename = './validation/malignant/SOB_M_DC-14-3909-200-018.png'
test_folder = './test'

images = []

for img in os.listdir(test_folder):
    img = os.path.join(test_folder, img)
    img = image.load_img(img, target_size=(IMG_W, IMG_H))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    images.append(img)

print(images)

# pred = model.predict(x)
#
# if pred >= 0.5:
#     print(pred, f'{filename} is malignant')
# else:
#     print(pred, f'{filename} is benign')