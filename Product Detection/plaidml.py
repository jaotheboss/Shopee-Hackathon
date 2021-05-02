# this is simply for leveraging on Apple's GPU because keras does not support AMD GPUs

import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

# preparing the test data
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    '/Users/jaoming/Active Projects/Shopee Challenge/Product Detection/shopee-product-detection-dataset/test',
    target_size = (299, 299),
    batch_size = 1, 
    interpolation = "bilinear",
    shuffle = False
)
file_names = [i[5:] for i in test_generator.filenames]

# preparing the model
model = models.load_model('42classXception_v1.h5')

# actual prediction and output
y_pred = model.predict_generator(test_generator)
y_pred_values = np.apply_along_axis(np.argmax, 1, y_pred)
final_predictions = pd.DataFrame({'filename': file_names, 'category': y_pred_values})
final_predictions.to_csv('final_predictions.csv', index = False)
