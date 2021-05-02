# Creating the Xception architecture model
def Xception(num_classes = 14, pooling = 'average'):
    """
    Building the Xception architecture

    Parameters:
        num_class : integer, default = 14
            The number of classes that this model would output

        pooling : string, default = 'average', {'average', 'max}
            The kind of global pooling that will occur before the softmax output layer

    Returns:
        output : Xception CNN model
    """
    # Block 1
    img_input = layers.Input(shape = (_HEIGHT, _WIDTH, _COLOR_P))
    x = layers.Conv2D(32, (3, 3), strides = (2, 2), use_bias = False)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(64, (3, 3), use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 1's Residual Layer
    residual = layers.Conv2D(128, (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(x)
    residual = layers.BatchNormalization()(residual)
    
    # Block 2
    x = layers.SeparableConv2D(128, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.SeparableConv2D(128, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)                             # Pool instead of Activation layer

	# Dual chanelling Block 2 with Block 1's Residual Layer
    x = layers.add([x, residual])

    # Block 2's Residual Layer
    residual = layers.Conv2D(256, (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(x)
    residual = layers.BatchNormalization()(residual)

    # Block 3
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.SeparableConv2D(256, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.SeparableConv2D(256, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)                             # Pool instead of Activation layer
    
    # Dual chanelling Block 3 with Block 2's Residual Layer
    x = layers.add([x, residual])

    # Block 3's Residual Layer
    residual = layers.Conv2D(512, (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(x)
    residual = layers.BatchNormalization()(residual)
    
    # Block 4
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.SeparableConv2D(512, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.SeparableConv2D(512, (3, 3), padding = 'same', use_bias = False)(x)                     # Pool instead of Activation layer
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
    
    # Dual chanelling Block 4 with Block 4's Residual Layer
    x = layers.add([x, residual])
    
    # Block 5 - 12
    for i in range(8):
        residual = x
        
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(512, (3, 3), padding = 'same', use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        x = layers.SeparableConv2D(512, (3, 3), padding = 'same', use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        x = layers.SeparableConv2D(512, (3, 3), padding = 'same', use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.add([x, residual])
        
    # Residual Layer for previous blocks
    residual = layers.Conv2D(768, (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(x)
    residual = layers.BatchNormalization()(residual)
    
    # Block 13
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.SeparableConv2D(512, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.SeparableConv2D(768, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
    
    # Dual chanelling Block 13 with Block 5-12's Residual Layer
    x = layers.add([x, residual])
    
    # Block 14
    x = layers.SeparableConv2D(1024, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 14 part 2
    x = layers.SeparableConv2D(2048, (3, 3), padding = 'same', use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Fully Connected Layer
    if pooling == 'average':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPool2D()(x)
    else:
        raise AttributeError('Pooling attribute value not recognised. Please call for either average or max.')
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(2048, activation = 'relu')(x)
    x = layers.Dense(num_classes, activation = 'softmax')(x)
    
    # Create model
    model = models.Model(img_input, x, name = 'Custom_Xception')
    
    return model
