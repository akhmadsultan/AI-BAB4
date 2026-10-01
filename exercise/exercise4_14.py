# Exercise 4.14
# Modify the Python program from Example 4.15,
# use an if-else statement to choose which classification model to use.

from keras.applications.imagenet_utils import decode_predictions
import keras
import numpy as np
import cv2

# Select model using if-else
model_name = 'mobilenetv2'  # Change this to switch models

if model_name == 'vgg16':
    from keras.applications.vgg16 import VGG16, preprocess_input
    clf = VGG16
    sz = 224

elif model_name == 'resnet50':
    from keras.applications.resnet50 import ResNet50, preprocess_input
    clf = ResNet50
    sz = 224

elif model_name == 'mobilenetv2':
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    clf = MobileNetV2
    sz = 224

elif model_name == 'densenet201':
    from keras.applications.densenet import DenseNet201, preprocess_input
    clf = DenseNet201
    sz = 224

elif model_name == 'inceptionv3':
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    clf = InceptionV3
    sz = 299

else:
    print("Unknown model name!")
    exit()


model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()

camera = cv2.VideoCapture(0)
image_size = sz

while True:

    ret, cam_frame = camera.read()

    if not ret:
        break

    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    preds = model.predict(image, verbose=0)

    label = decode_predictions(preds)

    cv2.putText(
        cam_frame,
        "{}, {:.1f}".format(
            label[0][0][1],
            label[0][0][2]
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Classification",
        cam_frame
    )

    key = cv2.waitKey(30)

    if key == 27:  # press 'ESC' to quit
        break


camera.release()
cv2.destroyAllWindows()