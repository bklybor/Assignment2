#! /usr/bin/python
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras import utils
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz

num_classes = 10
batch_size = 128
epochs = 20

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train/=255

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Dense(5, activation='linear', input_shape=(784,)))
    model.add(Dense(10, activation='linear', input_shape=(784,)))
    # model.add(Dense(10, activation='softmax', input_shape=(784,)))
    

    model.compile(loss = "mse", optimizer = SGD(lr = 0.50), metrics=['categorical_accuracy'])
    # plot_model(model, to_file='model.png')

    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    # ann_viz(model)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])