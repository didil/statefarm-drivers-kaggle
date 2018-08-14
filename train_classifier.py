import os, shutil, pathlib

full_data_dir = '/home/didil/Downloads/statefarm-drivers/imgs/train'

base_dir = './my-imgs'
pathlib.Path(base_dir).mkdir(exist_ok=True)

train_dir = os.path.join(base_dir, 'train')
pathlib.Path(train_dir).mkdir(exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
pathlib.Path(validation_dir).mkdir(exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
pathlib.Path(test_dir).mkdir(exist_ok=True)

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(230, 230, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        # All images will be resized to 230x230
        target_size=(230, 230),
        batch_size=50,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(230, 230),
        batch_size=50,
        class_mode='categorical')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=260,
      epochs=9,
      validation_data=validation_generator,
      validation_steps=60)

model.save('statefarm_drivers_4.h5')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(230, 230),
        batch_size=50,
        class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=60)
print('test_loss acc:', test_loss)
print('test acc:', test_acc)