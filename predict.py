from keras.models import load_model
import os
import numpy as np
from keras.preprocessing import image

model = load_model('statefarm_drivers_4.h5')

submission_data_dir = '/home/didil/Downloads/statefarm-drivers/imgs/test'

class_labels = [
    "normal driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger"
]

file_names = np.random.choice(os.listdir(submission_data_dir), 20)

img_arrays = []

for file_name in file_names:
    img = image.load_img(os.path.join(submission_data_dir, file_name), target_size=(230, 230))
    img_array = image.img_to_array(img)
    img_arrays.append(img_array)

img_arrays = np.array(img_arrays)
img_arrays = img_arrays.astype('float32') / 255
predictions = model.predict(img_arrays)

label_indexes = np.argmax(predictions, axis=1)
probabilities = np.max(predictions, axis=1)

for (file_name, label_index, probability) in zip(file_names, label_indexes, probabilities):
    if probability < 0.95:
        continue

    label_with_probability = "{}: {:.2f}%".format(class_labels[label_index], probability * 100)

    import cv2

    image = cv2.imread(os.path.join(submission_data_dir, file_name))

    cv2.putText(image, label_with_probability.upper(), (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite("annotated-results/" + file_name, image)
