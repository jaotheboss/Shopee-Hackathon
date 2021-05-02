import os
origin = os.getcwd()
data_source = '/Users/jaoming/Active Projects/Shopee Challenge/Product Detection/shopee-product-detection-dataset/additional'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import imgaug.augmenters as iaa

os.chdir(data_source)
datagen = ImageDataGenerator(rescale = 1/255,
                            rotation_range = 30, 
                            width_shift_range = 0.15,
                            height_shift_range = 0.35,
                            shear_range = 0.1,
                            zoom_range = 0.3,
                            horizontal_flip = False,
                            vertical_flip = True,
                            brightness_range = (0.8, 1.4))
image_gen = datagen.flow_from_directory('target', target_size = (299, 299), batch_size = 1, shuffle = True)
# x_name = image_gen.filenames[0]
# x = next(image_gen)[0][0]
# show image
# plt.imshow(x)

# saving the images
file_names = list(image_gen.filenames)
for i, im in enumerate(image_gen):
       imsave('e_' + file_names[i].split('/')[1], im[0][0])
       if i >= 196:
              break