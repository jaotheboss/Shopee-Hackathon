import os
origin = os.getcwd()
os.chdir('shopee-product-detection-dataset')

# modules for dataset preparation
import pandas as pd                                            # reading in the csv data
import cv2                                                     # for importing images
from sklearn.model_selection import train_test_split
import numpy as np

# image constants
_HEIGHT = 300
_WIDTH = 300
_COLOR_L = 3

# model constants
_BATCHSIZE = 5
_EPOCH = 5

def extract_images(dimension = (_HEIGHT, _WIDTH), n = 100, include = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']):
       """
       To extract the images out from the shopee-product-detection-dataset folder

       Parameters:
              dimension : tuple, default = (500, 500)
                     The dimension to resize all the images to (height, width)

              n : integer, default = 100
                     The number of images to extract per label
                     
              color : boolean, default = True
                     Whether to read in the image with RGB values or black and white (BW)

              include : list of strings, default = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
                     For specifying which folders you want to extract images from

       Returns:
              output : Two lists
                     First list to hold the array representation of the image, the second list for the labels
       """
       # establish directory routes
       origin = '/Users/jaoming/Active Projects/Shopee Challenge/Product Detection/shopee-product-detection-dataset'
       main_train_folder = '/Users/jaoming/Active Projects/Shopee Challenge/Product Detection/shopee-product-detection-dataset/train/train'
       os.chdir(main_train_folder)

       # setting up the variables 
       data, labels = [], []
       for name in include:
              os.chdir(name)
              image_namelist = os.listdir()
              if '.DS_Store' in image_namelist:                # removing unnecessary files
                     image_namelist.remove('.DS_Store')
              count = 0
              while count < n:
                     image = cv2.resize(
                            cv2.imread(image_namelist[count], cv2.IMREAD_COLOR),
                            dimension,
                            interpolation = cv2.INTER_CUBIC
                     )
                     data.append(image)
                     labels.append(int(name))
                     count += 1
              os.chdir(main_train_folder)

       os.chdir(origin)
       return data, labels

images, labels = extract_images(n = 10)
train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size = 0.2)
train_x, test_x, train_y, test_y = np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)

# modules for model creation (PlaidML)
# if we want to leverage the AMD GPU - https://github.com/plaidml/plaidml
# run plaidml-setup on terminal to set up which GPU plaidml should leverage
# import plaidml.keras
# plaidml.keras.install_backend()
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

# modules for model creation (Tensorflow backend for Keras)
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Nadam
from sklearn.metrics import accuracy_score

def shallow_vgg():
       model = models.Sequential()       
       model.add(layers.Conv2D(filters = 32,
              kernel_size = (3, 3),
              kernel_initializer = 'he_uniform', # or random_normal
              input_shape = (_HEIGHT, _WIDTH, _COLOR_L)))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))
       model.add(layers.MaxPooling2D((2, 2)))
       model.add(layers.Dropout(0.3))

       model.add(layers.Conv2D(filters = 64,
              kernel_size = (3, 3),
              kernel_initializer = 'he_uniform'))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))
       model.add(layers.MaxPooling2D((2, 2)))
       model.add(layers.Dropout(0.3))

       model.add(layers.Conv2D(filters = 128,
              kernel_size = (3, 3),
              kernel_initializer = 'he_uniform'))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))
       model.add(layers.MaxPooling2D((2, 2)))
       model.add(layers.Dropout(0.3))

       model.add(layers.Conv2D(filters = 256,
              kernel_size = (3, 3),
              kernel_initializer = 'he_uniform'))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))
       model.add(layers.MaxPooling2D((2, 2)))

       model.add(layers.GlobalAveragePooling2D())
       model.add(layers.Dropout(0.3))
       model.add(layers.Dense(256, activation = 'relu'))
       model.add(layers.Dense(14, activation = 'softmax'))
       return model
model = shallow_vgg()
model.summary()
# model.load_weights

os.chdir(origin)

sgd = SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)
nadam = Nadam(learning_rate = 0.002)

early_stopping = EarlyStopping(monitor = 'loss', 
       patience = 0, 
       min_delta = 0.001)

model_checkpoint_callback = ModelCheckpoint(
    filepath = 'Checkpoints/svggp_model-{epoch:02d}-{acc:.4f}.hdf5',
    monitor = 'acc',
    mode = 'max',
    save_best_only = True)

model.compile(optimizer = nadam,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc'])

# data filter. rotates/flips/manipulates image that's passed through here
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 30, 
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_generator = train_datagen.flow(train_x, train_y, batch_size = _BATCHSIZE)

history = model.fit_generator(train_generator,
                            steps_per_epoch = len(train_y)/_BATCHSIZE,                      # so the model trains per batch
                            epochs = _EPOCH,
                            callbacks = [early_stopping, model_checkpoint_callback])

y_pred = np.argmax(model.predict(test_x), axis = 1)
accuracy_score(test_y, y_pred)
