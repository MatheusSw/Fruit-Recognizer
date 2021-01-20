import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from glob import glob

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

training_dataset_directory = r"C:\Users\wordp\Desktop\Fruit recognizer\fruits-360\Training"
test_dataset_directory = r"C:\Users\wordp\Desktop\Fruit recognizer\fruits-360\Test"

image_size = (100, 100)
batch_size = 64
epochs = 5
seed = 53

# Refactor this
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.5, horizontal_flip=True, validation_split=0.3)
train_generator = train_datagen.flow_from_directory(
    target_size=image_size, batch_size=batch_size, directory=training_dataset_directory)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.5, horizontal_flip=True, validation_split=0.3)
valid_generator = valid_datagen.flow_from_directory(
    target_size=image_size, batch_size=batch_size, directory=training_dataset_directory)

 model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                           input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Drop
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(
        len(train_generator.class_indices), activation='softmax')
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=.0020, patience=2, verbose=0,
    mode='min', baseline=None, restore_best_weights=False
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=epochs, validation_data = valid_generator, verbose=1, callbacks=[early_stopping])

model.save('models/trained_less_augmentation_more_filters')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
