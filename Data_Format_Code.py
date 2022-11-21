import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
label = ['cracks', 'no_cracks']

directory = "/Users/kageammons/Desktop/archive/Bridge_Crack_Image/DBCC_Training_Data_Set/train" #Change this to match your download location
labels = ['cracked','non-cracked']

for image in os.listdir(directory):
    image_array = cv2.imread(os.path.join(directory,image), cv2.IMREAD_GRAYSCALE)
    # plt.imshow(image_array, cmap='gray')
    break
   
common_scale = 15
new_image_array = cv2.resize(image_array, (common_scale, common_scale))
plt.imshow(new_image_array, cmap='gray')
plt.show()

# Note, there are 6000 images of cracks and 44000 images of no_cracks for a total of 50,000 training images
# Cracks = 0
# No_Cracks = 1
common_scale = 15
training_array = []

def create_training_array():
    iterator = 1
    for image in os.listdir(directory):
        if iterator < 6001:
            try:
                class_num = 0 # Cracked or defective
                image_array = cv2.imread(os.path.join(directory,image), cv2.IMREAD_GRAYSCALE)
                new_image_array = cv2.resize(image_array, (common_scale, common_scale))
                training_array.append([new_image_array, class_num])
            except Exception as e:
                pass
        else:
            try:
                class_num = 1 # Stable, nondefective
                image_array = cv2.imread(os.path.join(directory,image), cv2.IMREAD_GRAYSCALE)
                new_image_array = cv2.resize(image_array, (common_scale, common_scale))
                training_array.append([new_image_array, class_num])
            except Exception as e:
                pass
        iterator = iterator + 1

create_training_array()            

X = []
y = []
for features, label in training_array:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, common_scale, common_scale, 1)

# This is necessary to save the data
pickle_out = open("Xbridge.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("ybridge.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

from tensorflow import keras
model.save('CNN_model')
