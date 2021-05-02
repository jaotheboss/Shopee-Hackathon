import cv2       
import os 

def extract_images(dimension = (_HEIGHT, _WIDTH), n = 100, color = True, include = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']):
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
       origin = '/Users/jaoming/Active Projects/Shopee Challenge/shopee-product-detection-dataset'
       main_train_folder = '/Users/jaoming/Active Projects/Shopee Challenge/shopee-product-detection-dataset/train/train'
       os.chdir(main_train_folder)
       if color:
              imread_color = cv2.IMREAD_COLOR
       else:
              imread_color = cv2.IMREAD_GRAYSCALE

       # setting up the variables 
       data, labels = [], []
       for name in include:
              os.chdir(name)
              image_namelist = os.listdir()
              if '.DS_Store' in image_namelist:                # removing unnecessary files
                     image_namelist.remove('.DS_Store')
              count = 0
              while count < n:
                     data.append(cv2.resize(
                            cv2.imread(image_namelist[count], imread_color),
                            dimension,
                            interpolation = cv2.INTER_CUBIC
                     ))
                     labels.append(int(name))
                     count += 1
              os.chdir(main_train_folder)

       os.chdir(origin)
       return data, labels
