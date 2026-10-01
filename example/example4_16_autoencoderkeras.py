# Example 4.16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.datasets import mnist
import numpy as np

# Load dataset
(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

# Normalize data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

IMG_SHAPE = x_train.shape[1:]


def build_autoencoder(img_shape, encoded_dim):

    # Encoder
    encoder = Sequential()
    encoder.add(Flatten(input_shape=img_shape))
    encoder.add(Dense(encoded_dim, activation='relu'))

    # Decoder
    decoder = Sequential()
    decoder.add(Dense(128, input_dim=encoded_dim, activation='relu'))

    # Perbaikan Keras 3:
    decoder.add(Dense(int(np.prod(img_shape))))

    decoder.add(Reshape(img_shape))

    return encoder, decoder


encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

# Build autoencoder
autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

# Compile
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Summary
autoencoder.summary()

# Training
autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)