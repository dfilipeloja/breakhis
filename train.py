from keras_preprocessing.image import ImageDataGenerator
from models import alexnet

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('train/',
                                          target_size=(32, 32),
                                          batch_size=32,
                                          class_mode='binary')

test_dataset = train.flow_from_directory('test/',
                                          target_size=(32, 32),
                                          batch_size=32,
                                          class_mode='binary')

#print(test_dataset.class_indices)

model = alexnet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_dataset,
                    steps_per_epoch=250,
                    epochs=10,
                    validation_data=test_dataset)
