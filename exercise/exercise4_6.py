# Exercise 4.6
# Modify the Python program from Example 4.7, and add another Conv2D, MaxPooling, and Dropout layer before the Flatten layer. Compare the number of parameters.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(
    32,
    (5, 5),
    input_shape=(28, 28, 1),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Added Conv2D, MaxPooling, and Dropout layer before Flatten
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())