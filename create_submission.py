import os

submission_data_dir = '/home/didil/Downloads/statefarm-drivers/imgs/test'

from keras.models import load_model

model = load_model('statefarm_drivers_4.h5')

test_file_names = os.listdir(submission_data_dir)

import numpy as np
from keras.preprocessing import image

batches = np.array_split(np.array(test_file_names), 230)

results = [['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']]

for batch_index, batch_file_names in enumerate(batches):
    print("batch index: " + str(batch_index))
    batch_img_arrays = []
    for file_name in batch_file_names:
        img = image.load_img(os.path.join(submission_data_dir, file_name), target_size=(230, 230))
        img_array = image.img_to_array(img)
        batch_img_arrays.append(img_array)

    batch_img_arrays = np.array(batch_img_arrays)
    batch_img_arrays = batch_img_arrays.astype('float32') / 255
    predictions = model.predict(batch_img_arrays)
    # print("predictions:")
    # print(batch_file_names[0] )
    # print(predictions[0] )
    # break

    for (file_name, prediction) in zip(batch_file_names, predictions):
        results.append([file_name, *prediction])

#for index, result in enumerate(results):
#    if(index !=0):
#        for j in range(1,11):
#            result[j] =  "{:.5f}".format(float(result[j]))

import csv

with open('submission.csv', 'w') as submission_file:
    wr = csv.writer(submission_file)
    for result in results:
        wr.writerow(result)
