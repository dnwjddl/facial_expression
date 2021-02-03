import numpy as np
import pandas as pd
import os
#print(os.listdir("./data/"))


import pandas as pd
import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


file = './data/fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names = ['emotion', 'pixels','usage']
df = pd.read_csv(file, names = names, na_filter = False)
im = df['pixels']

def getData(file):
    # images are 48x4
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(file):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X,Y = np.array(X)/255.0, np.array(Y)
    return X,Y

X, Y = getData(file)
num_class = len(Y)

N, D = X.shape
X = X.reshape(N, 48, 48, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)


def model():
    model = Sequential()
    input_shape = (48, 48, 1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    # model.summary()

    return model

model = model()
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

path_model='model.h5' # save model at this location after each epoch

# # K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
# model=model() # create the model
# K.set_value(model.optimizer.lr,1e-3) # set the learning rate

batch_size = 64
# fit the model
h=model.fit(x=X_train, y=y_train,
            batch_size=64, epochs=20,
            verbose=1,
            validation_data=(X_test,y_test),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])