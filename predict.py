import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from datetime import datetime

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

y_true = gen.classes

for filename in gen.filenames:
    img = image.load_img(
            test_dir + '/' + filename,
            target_size=(400, 400))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    images.append(img)

images = np.vstack(images)

start = datetime.now()
print('Start', start)

y_pred = model.predict(images)

y_pred = (y_pred > 0.5)
y_pred = np.vstack(y_pred)
#
cm = confusion_matrix(y_true, y_pred)

end = datetime.now()
print('End', end)

fig, ax = plt.subplots()
plt.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.title('Confusion Matrix Recognition')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig('results/confusion_matrix1.png')
