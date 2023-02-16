import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import numpy as np
from PIL import Image
from skimage import transform


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (28, 28, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    y_train = to_categorical(y_train)
    cnn = Sequential()
    cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, 'relu'))
    cnn.add(Dense(10, 'softmax'))
    cnn.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)


def predict_wrong():
    figure, axes = plt.subplots(4, 6, figsize=(16, 9))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_test = to_categorical(y_test)
    przypuszczenia = cnn.predict(x_test)
    for i, p in enumerate(przypuszczenia[0]):
        print(f'{i}: {p:.10%}')
    obrazy = x_test.reshape((10000, 28, 28, 1))
    chybione_prognozy = []
    for i, (p, e) in enumerate(zip(przypuszczenia, y_test)):
        prognozowany, spodziewany = np.argmax(p), np.argmax(e)
        if prognozowany != spodziewany:
            chybione_prognozy.append((i, obrazy[i], prognozowany, spodziewany))
    print(len(chybione_prognozy))
    for axes, element in zip(axes.ravel(), chybione_prognozy):
        axes.imshow(element[1], cmap=plt.cm.gray_r)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(f'{element[0]}\np:{element[2]};, s:{element[3]}')
    plt.tight_layout()
    cnn.save('model')
    plt.show()


cnn = load_model('model')
predict_wrong()
img = load('img.png')
xd = cnn.predict(img)
print("found ", np.argmax(xd))
