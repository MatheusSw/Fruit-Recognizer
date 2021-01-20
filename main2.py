import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from glob import glob

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

test_dataset_directory = r"C:\Users\wordp\Desktop\Fruit recognizer\fruits-360\Test"

image_size = (100, 100)
batch_size = 64
epochs = 5
seed = 53

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dataset_directory,image_size=image_size,
)

def base(path):
    return os.path.basename(path)

class_names = glob("fruits-360/Training/*")  # Reads all the folders in which images are present
class_names = list(sorted(map(base, class_names)))  # Sorting them

model = tf.keras.models.load_model('models/trained')


img_path = tf.keras.utils.get_file('eu', origin="https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg")
img = tf.keras.preprocessing.image.load_img(
    r'C:\Users\wordp\Desktop\apricot.jpg', target_size=image_size
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(predictions[0])], 100 * np.max(predictions[0]))
)