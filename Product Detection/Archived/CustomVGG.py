import os
origin = os.getcwd()

# CONSTANTS
_HEIGHT = 300
_WIDTH = 300      
_COLOR = 3     

_KERNEL_SIZE = (3, 3)
_NUM_CLASSES = 14
_BATCHSIZE = 8
_EPOCHS = 1

# modules for dataset preparation
from keras.preprocessing.image import ImageDataGenerator

# Settling the data
# data filter. rotates/flips/manipulates image that's passed through here
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 30, 
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(
    'shopee-product-detection-dataset/train/pd_train', 
    target_size = (_HEIGHT, _WIDTH),
    batch_size = _BATCHSIZE, 
    interpolation = "bilinear",
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    'shopee-product-detection-dataset/train/pd_train',
    target_size = (_HEIGHT, _WIDTH),
    batch_size = _BATCHSIZE,
    interpolation = "bilinear",
    subset = 'validation'
)

# modules for model creation
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adadelta, Adam
from sklearn.metrics import accuracy_score

# Settling the model
# Importing the original Xception architecture and adding a customized head
def CustomVGG(num_classes = 14, pooling = 'average'):
       """
       Setting up a customized vgg architecture cnn model

       Parameters:
              num_class : integer, default = 14
              The number of classes that this model would output

              pooling : string, default = 'average', {'average', 'max}
              The kind of global pooling that will occur before the softmax output layer

       Returns:
              output : VGG architecture CNN model
       """
       model = models.Sequential()

       # Block 1
       model.add(layers.Conv2D(filters = 32,
              kernel_size = _KERNEL_SIZE,
              activation = 'relu',
              input_shape = (_HEIGHT, _WIDTH, _COLOR)))

       model.add(layers.Conv2D(filters = 32,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       ## Block 1 - Regularisation
       model.add(layers.MaxPooling2D())
       #model.add(layers.SpatialDropout2D(0.3))


       # Block 2
       model.add(layers.Conv2D(filters = 64,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       model.add(layers.Conv2D(filters = 64,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       ## Block 2 - Regularisation
       model.add(layers.MaxPooling2D())
       #model.add(layers.SpatialDropout2D(0.3))


       # Block 3
       model.add(layers.Conv2D(filters = 128,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       model.add(layers.Conv2D(filters = 128,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       # model.add(layers.Conv2D(filters = 128,
       #        kernel_size = _KERNEL_SIZE))
       # model.add(layers.BatchNormalization())
       # model.add(layers.Activation('relu'))
       
       ## Block 3 - Regularisation
       model.add(layers.MaxPooling2D())
       #model.add(layers.SpatialDropout2D(0.3))


       # Block 4
       model.add(layers.Conv2D(filters = 256,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       model.add(layers.Conv2D(filters = 256,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       # model.add(layers.Conv2D(filters = 256,
       #        kernel_size = _KERNEL_SIZE))
       # model.add(layers.BatchNormalization())
       # model.add(layers.Activation('relu'))
       
       ## Block 4 - Regularisation
       model.add(layers.MaxPooling2D())
       #model.add(layers.SpatialDropout2D(0.3))


       # Block 5
       model.add(layers.Conv2D(filters = 512,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       model.add(layers.Conv2D(filters = 512,
              kernel_size = _KERNEL_SIZE))
       model.add(layers.BatchNormalization())
       model.add(layers.Activation('relu'))

       # model.add(layers.Conv2D(filters = 512,
       #        kernel_size = _KERNEL_SIZE))
       # model.add(layers.BatchNormalization())
       # model.add(layers.Activation('relu'))


       # Fully Connected Layer
       if pooling == 'average':
              model.add(layers.GlobalAveragePooling2D())
       elif pooling == 'max':
              model.add(layers.GlobalMaxPooling2D())
       else:
              raise AttributeError('Pooling attribute value not recognised. Please call for either average or max.')
       model.add(layers.Dropout(0.3))
       model.add(layers.Dense(512, activation = 'relu'))
       model.add(layers.Dense(num_classes, activation = 'softmax'))

       return model

model = CustomVGG()
model.summary()

rmsprop = RMSprop(learning_rate = 0.001, rho = 0.85)            
adadelta = Adadelta(learning_rate = 0.8, rho = 0.95)      
adam = Adam(learning_rate = 0.001)

early_stopping = EarlyStopping(monitor = 'val_loss', 
       patience = 0, 
       min_delta = 0.01)

model_checkpoint_callback = ModelCheckpoint(
    filepath = 'Checkpoints/model-{epoch:02d}-{val_acc:.4f}.hdf5',
    monitor = 'val_acc',
    mode = 'max',
    save_best_only = True)

model.compile(optimizer = adam,
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.n//_BATCHSIZE,                   
    validation_data = validation_generator,
    validation_steps = validation_generator.n//_BATCHSIZE,
    epochs = _EPOCHS,
    callbacks = [early_stopping, model_checkpoint_callback]
)