# Exercise 4.4

# Modify Python LeNet program from the Example 4.4 to use a different number of filters and see how this affects the total number of parameters.

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers

model = Sequential()

model.add(Conv2D(
    filters=12,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(32, 32, 1)
))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu'
))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.summary()