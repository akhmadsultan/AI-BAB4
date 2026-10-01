#Exercise 4.17
#Modify the Python program from Example 4.17, select n=2, n=4, or n=5 layer to visualize the feature maps. Comment on your results.

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

model = VGG19()

# change n to visualize different layers: n=2, n=4, or n=5
n = 2
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
feature_maps = model.predict(img)
print("Feature maps: ", feature_maps.shape)

def plot_feature_maps(feature_maps):
 col = 8
 row = int(feature_maps.shape[3]/col)
 ix = 1
 plt.figure(figsize=(20,20))
 for _ in range(row):
  for _ in range(col):
   ax = plt.subplot(row, col, ix)
   ax.set_xticks([])
   ax.set_yticks([])
   plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
   ix += 1
 plt.show()

plot_feature_maps(feature_maps)
