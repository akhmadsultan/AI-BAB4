#Exercise 4.16
#Modify the Python program from Example 4.17, change the n_filters value to display all the filters.

from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape 
print("Color channels: ", s[0])
print("Filter size: ", s[1],s[2])
print("Total number of filters : ", s[3])

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# change n_filters to display all the filters
n_filters = s[3]
ix = 1

pyplot.figure(figsize=(20,20))
for i in range(n_filters):
 f = filters[:, :, :, i]
 for j in range(s[0]):
  ax = pyplot.subplot(n_filters, s[0], ix)
  ax.set_xticks([])
  ax.set_yticks([])
  pyplot.imshow(f[j,:, :], cmap='gray')
  ix += 1
pyplot.show()
