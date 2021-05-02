# defining the constants of the project
# FOR DATA #
_HEIGHT = 299      
_WIDTH = 299       
_COLOR = True       
_NUMCLASSES = 14    

# FOR MODEL #
_BATCHSIZE = 32          # the number of images to train at each 'iteration' within an epoch
_EPOCHS = 10
if _COLOR:
       _COLOR_P = 3     
else:
       _COLOR_P = 1
_WEIGHTS_NAME = 'model-01-0.7210.hdf5'

# importing modules
import os
output_dir = os.getcwd()
# importing modules that builds the model
from keras.applications.xception import Xception
from keras import layers, models
from keras.optimizers import RMSprop, Adadelta, SGD, Adamax, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

# passing the data through a data filter. rotates/flips/manipulates image that's passed through here
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 30, 
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/product-detection-images-first-14/pd_train', 
    target_size = (_HEIGHT, _WIDTH),
    batch_size = _BATCHSIZE, 
    interpolation = "bilinear",
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    '/kaggle/input/product-detection-images-first-14/pd_train',
    target_size = (_HEIGHT, _WIDTH),
    batch_size = _BATCHSIZE,
    interpolation = "bilinear",
    subset = 'validation'
)

# Importing the original Xception architecture without the head - 1 EPOCH is 0.7102
def TransferXception(num_classes = 14, pooling = 'average', imagenet_weights = True):
    """
    Setting up the Keras Xception application model architecture with a 
    customizable head

    Parameters:
        num_class : integer, default = 14
            The number of classes that this model would output

        pooling : string, default = 'average', {'average', 'max}
            The kind of global pooling that will occur before the softmax output layer

    Returns:
        output : Xception CNN model
    """
    if imagenet_weights:
        w = 'imagenet'
    else:
        w = None
    img_input = layers.Input(shape = (_HEIGHT, _WIDTH, _COLOR_P))
    custom_xception = Xception(include_top = False, weights = '/kaggle/input/inceptionresnetv2/xception_tf_notop.h5', input_tensor = img_input)
    for layer in custom_xception.layers[:129]:
      layer.trainable = False
    
    x = custom_xception.layers[-1].output # 3rd layer from the last, block14_sepconv2
    
    ### Decoder
    if pooling == 'average':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool2D()(x)
    else:
        raise AttributeError('Pooling attribute value not recognised. Please call for either average or max.')
    x = layers.Dropout(0.3)(x)
    # Original - 7,388,686 trainable parameters
    #x = layers.Dense(2048, activation = 'relu')(x)

    # Deep - 6,325,774 trainable parameters
    x = layers.Dense(2048, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation = 'relu')(x)

    # Wide - 11,613,710 trainable parameters
    #x = layers.Dense(4096, activation = 'relu')(x)
    output = layers.Dense(num_classes, activation = 'softmax')(x)
    
    model = models.Model(inputs = custom_xception.input, outputs = output, name = 'Transfer_Xception')

    return model

os.chdir(output_dir)

# crafting the Xception model
model = TransferXception(imagenet_weights = False)
model.load_weights('model-01-0.7918.hdf5')

# previewing the architecture of the resultant model
model.summary()

# compiling and setting up the loss function, optimizer and metrics of the model
sgd = SGD(learning_rate = 0.01, momentum = 0.0, nesterov = False)               # better for shallower networks
# adagrad is better with sparse data (not really good for images)
rmsprop = RMSprop(learning_rate = 0.001, rho = 0.85)                            
adadelta = Adadelta(learning_rate = 1.0, rho = 0.95)      
adam = Adam(learning_rate = 0.001)
adamax = Adamax(learning_rate = 0.001)

early_stopping = EarlyStopping(monitor = 'val_loss', 
       patience = 1, 
       min_delta = 0.01)

model_checkpoint_callback = ModelCheckpoint(
    filepath = output_dir + '/model-{epoch:02d}-{val_acc:.4f}.hdf5',
    monitor = 'val_acc',
    mode = 'max',
    save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 0)

model.compile(optimizer = adamax,
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.n//_BATCHSIZE,                      # so the model trains per batch
    validation_data = validation_generator,
    validation_steps = validation_generator.n//_BATCHSIZE,
    epochs = _EPOCHS,
    callbacks = [early_stopping, model_checkpoint_callback, reduce_lr]
)
