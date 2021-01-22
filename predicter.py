import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from glob import glob

def base(path):
    return os.path.basename(path)

image_size = (100, 100)

#Switch this with a generator?
class_names = glob("fruits-360/Training/*")
class_names = list(sorted(map(base, class_names)))

model = tf.keras.models.load_model('models/trained_increased_kernel_and_neurons')

img = tf.keras.preprocessing.image.load_img(
    r'C:\Users\wordp\Desktop\avocado.jfif', target_size=image_size
)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)


predictions = model.predict(img_array)

top_k_values, top_k_indices = tf.nn.top_k(predictions, k=5)
top_k_indices, top_k_values = (top_k_indices.numpy()[0], top_k_values.numpy()[0])

for idx in range(len(top_k_values)):
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[top_k_indices[idx]], 100 * np.max(top_k_values[idx]))
    )
