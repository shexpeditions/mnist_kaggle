
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Dense
from keras.layers.normalization import BatchNormalization

def conv_net(input_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(Conv2D(32, kernel_size=(5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(Conv2D(32, kernel_size=(5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Dropout(0.3))
    model.add(Flatten())
    
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization());
    # model.add(Dropout(0.3))

    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization());
    # model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization());
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    return model