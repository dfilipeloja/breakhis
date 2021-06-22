import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image

model = load_model('./models/breakhis_vgg19_model.h5')

model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy'])

filename = './validation/benign/SOB_B_A-14-29960CD-200-007.png'

img = image.load_img(
        filename,
        target_size=(400, 400))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

pred = model.predict(x)

if pred >= 0.5:
    print(pred, f'{filename} is malignant')
else:
    print(pred, f'{filename} is benign')