import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
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

treshhold = 10

y_pred = (y_pred > (treshhold / 100))
y_pred = np.vstack(y_pred)

now = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

cm = confusion_matrix(y_true, y_pred)
classification_report = classification_report(y_true, y_pred, target_names=['benign', 'malignant'])

with open("results/classification_reports/classification_report_" + now + "_th_" + str(treshhold) + ".txt", "w") as text_file:
    print(classification_report, file=text_file)

fig, ax = plt.subplots()
ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
ax.set_title('Confusion Matrix Recognition')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicated Label')
plt.savefig('results/confusion_matrix_' + now + '_th_' + str(treshhold) + '.png')
