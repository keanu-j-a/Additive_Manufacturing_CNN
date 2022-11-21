# This quick notebook will analyze if the model produce for Mr. Deb is able to correctly identify defects
import cv2
import tensorflow as tf

CATEGORIES = ["Defective","Non-Defective"]
location = "/Users/kageammons/Desktop/CNN_Defect_Tool"

def prepare(location):
    img_sizes = 15
    img_array = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_sizes,img_sizes))
    return new_array.reshape(-1,img_sizes,img_sizes, 1)

model = tf.keras.models.load_model(location)

prediction = model.predict([prepare('/Users/kageammons/Desktop/CNN_Defect_Tool/crack0.jpg')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('/Users/kageammons/Desktop/CNN_Defect_Tool/crack1.jpg')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('/Users/kageammons/Desktop/CNN_Defect_Tool/nonDefective0.jpg')])
print(CATEGORIES[int(prediction[0][0])])
