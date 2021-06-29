import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

y_pred = model.predict(images)

y_pred = (y_pred > 0.5)
y_pred = np.vstack(y_pred)

start = datetime.now()
print('Start', start)

cm = confusion_matrix(model, y_true, y_pred)

end = datetime.now()
print('End', end)

ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['benign', 'malignant'])
ax.yaxis.set_ticklabels(['malignant', 'benign'])

ax.savefig("./results/vgg_confusion_matrix.png")
#
# print(classes)
#
# pred = model.predict(x)
#
# if pred >= 0.5:
#     print(pred, f'{filename} is malignant')
# else:
#     print(pred, f'{filename} is benign')