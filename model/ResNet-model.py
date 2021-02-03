''' 1. import libraries'''
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.datasets import cifar10
from keras.utils.data_utils import Sequence



''' 2. get the data'''
file = './data/fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names = ['emotion', 'pixels','usage']
df = pd.read_csv(file, names = names, na_filter = False)
im = df['pixels']

def getData(filname):
    # images are 48x48
    # N = 10000
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

X, Y = getData(file)
num_class = len(set(Y))
X = X*255
X = X[:1500]
Y = Y[:1500]


def pca_transform():
    standardized_data = StandardScaler().fit_transform(X)
    print(standardized_data.shape)
    # configuring the parameteres
    # the number of components = 2
    pca = decomposition.PCA()
    pca.n_components = 2
    pca_data = pca.fit_transform(standardized_data)

    # pca_reduced will contain the 2-d projects of simple data
    print("shape of pca_reduced.shape = ", pca_data.shape)
    pca_data = np.vstack((pca_data.T, Y)).T

    # creating a new data fram which help us in ploting the result data
    pca_df = pd.DataFrame(data=pca_data, columns=('1st_Dim', '2nd_Dim', "label"))
    sn.FacetGrid(pca_df, hue="label", height=5).map(plt.scatter, '1st_Dim', '2nd_Dim').add_legend()
    plt.show()


#pca_transform()

''' 3.  Keras with tensorflow backend'''
N, D = X.shape
print(X.shape) # (1500, 2)
X = X.reshape(N, 48, 48, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=3)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
input_shape = X_train.shape[1:]

''' 4. Defining the Resnet Model'''
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
n = 3
depth = 20


# Training parameters
batch_size = 32
epochs = 200
data_augmentation = True
num_classes = 7


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-5))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=7):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v1(input_shape=input_shape, depth=depth)
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

filepath = './resnet_model_filter_450.h5'
if not data_augmentation:
    print("Not using data augmentation")
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs,
              validation_data=(X_test, y_test), shuffle=True, callbacks = [ModelCheckpoint(filepath = filepath)])
else:
    print("Using real-time data augmentation")
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center= False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-03,
        rotation_range=0,
        width_shift_range=0.2,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval= 0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale = None,
        data_format=None,
        validation_split=0.0)
    datagen.fit(X_train)
    h = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 128,
                        validation_data=(X_test, y_test),
                        epochs=450, verbose=1, workers=4,
                        callbacks=[ModelCheckpoint(filepath=filepath)])
# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])