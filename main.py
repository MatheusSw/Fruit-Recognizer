import tensorflow as tf
import datetime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

training_dataset_directory = r"/home/wordp/fruits-360/Training"
validation_dataset_directory = r"/home/wordp/fruits-360/Validation"


def get_training_generator():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, rotation_range=3, shear_range=0.1, validation_split=0.5)

    return train_datagen.flow_from_directory(
        target_size=image_size, batch_size=batch_size, directory=training_dataset_directory)


def get_validation_generator():
    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, rotation_range=3, shear_range=0.1, validation_split=0.5)

    return validation_image_generator.flow_from_directory(
        target_size=image_size, batch_size=batch_size, directory=validation_dataset_directory)


def make_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               input_shape=(100, 100, 3), padding='same', name="conv2d_1"),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(4, 4, name="max_pool_2"),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
        tf.keras.layers.MaxPooling2D(2, 2, name="max_pool_3"),

        tf.keras.layers.Conv2D(256, (6, 6), activation='relu', name="conv2d_4"),
        tf.keras.layers.MaxPooling2D(2, 2, name="max_pool_4"),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv2d_5"),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(
            len(training_generator.class_indices), activation='softmax')
    ])


image_size = (100, 100)
batch_size = 64
epochs = 12

training_generator = get_training_generator()
validation_generator = get_validation_generator()

model = make_model()
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=.0045, patience=1, verbose=0,
    mode='min', baseline=None, restore_best_weights=False
)

log_dir = f'logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_generator, epochs=epochs,
                    validation_data=validation_generator, verbose=1, callbacks=[early_stopping, tensorboard_callback])

print("Saving model")
model.save('models')
print("Model saved")
