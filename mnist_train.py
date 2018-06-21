
import keras
import keras.datasets.mnist
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mnist_conv_nets

def scale_data_and_convert_to_float(data_x):
    data_x = data_x / 255.0
    data_x = data_x.astype('float32')
    return data_x

def one_hot_encode_labels(data_y):

    data_y = LabelBinarizer().fit_transform(data_y)
    data_y = data_y.astype('float32')

    return data_y

def define_data_generator():
    data_generator = ImageDataGenerator(rotation_range=3,
    width_shift_range=0.1,
    height_shift_range=0.1,    
    shear_range=2,
    zoom_range=0.2)

    return data_generator

def visualize_curves(history, filename):
    pp = PdfPages(filename)
    plt.style.use('bmh')
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.savefig(pp, format='pdf')

    plt.figure()
    
    plt.plot(history.history['acc'], label='train_acc')
    plt.plot(history.history['val_acc'], label='val_acc')

    plt.savefig(pp, format='pdf')

    pp.close()


def train(train_x, train_y, model_name, input_shape=(28, 28, 1), epochs = 100):
    
    fn_metric_log = '{0}_conv_net_.csv'.format(model_name)

    train_y = one_hot_encode_labels(train_y)
    #test_y = one_hot_encode_labels(test_y)

    kfold = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=4711)
    #kfold = StratifiedKFold(n_splits=3, random_state=4711)
        
    batch_size = 64
    for index, (train_indices, valid_indices) in enumerate(kfold.split(train_x, train_y)):

        # train_indices = split[0]
        # valid_indices = split[1]

        fn_best_model = 'best_{0}_conv_net_fold_{1}_with_reduce.hdf5'.format(model_name, index)

        print('Training samples:', len(train_indices))
        print('Validatiaon samples:', len(valid_indices))

        training_set_x = train_x[train_indices]
        trainset_set_y = train_y[train_indices]

        validation_set_x = train_x[valid_indices]
        validation_set_y = train_y[valid_indices]
        
        model = mnist_conv_nets.conv_net(input_shape=(28,28,1))
                
        model_ckpt = ModelCheckpoint(fn_best_model, monitor='val_loss', verbose=1, save_best_only=True)
        model_metric_logger = CSVLogger(fn_metric_log, ';')
        modelreduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.9)

        model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

        data_gen = define_data_generator()

        history = model.fit_generator(data_gen.flow(training_set_x, trainset_set_y, batch_size=batch_size),
                        steps_per_epoch=len(training_set_x) / batch_size, epochs=epochs, verbose=1,
                        validation_data=(validation_set_x, validation_set_y), callbacks=[model_ckpt, model_metric_logger, modelreduce_lr])

        visualize_curves(history, fn_best_model + '.pdf')

def test(test_x):
    fn_best_model = 'best_mnist_conv_net.hdf5'    

    #score = model.evaluate(test_x, test_y, verbose=0)
    model = mnist_conv_nets.conv_net(input_shape=(28, 28, 1))     
    
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    model.load_weights(fn_best_model)
    predicted_classes = model.predict_classes(test_x)
        
    return predicted_classes

def eval(test_x, test_y):
    fn_best_model = 'best_mnist_fashion_conv_net_fold_0.hdf5'
    
    #score = model.evaluate(test_x, test_y, verbose=0)
    model = mnist_conv_nets.conv_net(input_shape=(28, 28, 1))     
    
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    model.load_weights(fn_best_model)

    predictions = model.predict(test_x)
    score = model.evaluate(test_x, test_y)

    print('Test loss: ', score[0])
    print('Test accuracy:', score[1])

    print('classification report')
    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1)))

    print('confusion matrix')
    print(confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1)))

    return score[0]

def prepare_mnist_digist(filename, has_labels):
    df = pd.read_csv(filename, sep=',', header='infer')
    values = df.values

    if has_labels == True:
        y = values[:, 0]
        x = values[:, 1:]
        return x, y
    else:
        x = values[:, 0:]
        return x, None

if __name__ == '__main__':
    
    # Test accuracy: 0.9443 fashion mnist
    # Test accuracy: 0.996 mnist kaggle

    epochs = 3
    batch_size = 64

    # import mnist kaggle
    model_name = 'mnist_fashion_tmp'   
    #fn_train = 'J:/udacity/deeplearning/datasets/mnist_kaggle/train.csv'
    #fn_test = 'J:/udacity/deeplearning/datasets/mnist_kaggle/test.csv'

    fn_train = 'J:/udacity/deeplearning/datasets/mnist_fashion/train.csv'
    fn_test = 'J:/udacity/deeplearning/datasets/mnist_fashion/test.csv'

    train_x, train_y = prepare_mnist_digist(fn_train, True)
    test_x, test_y = prepare_mnist_digist(fn_test, True)

    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    print('Training X shape: {0}'.format(train_x.shape))
    print('Training Y shape: {0}'.format(train_y.shape))
    
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    print('Test X shape: {0}'.format(test_x.shape))

    #(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    print('Training shape: {0}'.format(train_x.shape))
    print('Test shape: {0}'.format(test_x.shape))
    
    # expand to 4 dimensions
    # train_x = np.expand_dims(train_x, -1)
    # test_x = np.expand_dims(test_x, -1)

    print('Training shape: {0}'.format(train_x.shape))
    print('Test shape: {0}'.format(test_x.shape))
    
    train_x = scale_data_and_convert_to_float(train_x)
    test_x = scale_data_and_convert_to_float(test_x)
    training = False
    if training == True:
        print('mnist trainer')
        train(train_x, train_y, model_name=model_name, epochs=epochs)
        
    else:
        print('mnist tester')
        # model.fit(training_set_x, trainset_set_y, batch_size=batch_size, epochs=epochs, verbose=1,
        #     validation_data=(validation_set_x, validation_set_y), callbacks=[model_ckpt, model_metric_logger])
        test_y = one_hot_encode_labels(test_y)
        eval(test_x, test_y)
        # fn_submission_file = 'submission_augmented_.csv'        
        # predicted_classes = test(test_x)
        # indices = np.array(list(range(1, len(predicted_classes)+1)))
        # submissionFrame = pd.DataFrame(data={'ImageId' : indices.astype(np.uint32), 'Label' : predicted_classes.astype(np.uint32)})        
        # submissionFrame.to_csv(fn_submission_file, index=False, index_label=False)     
        
        # print('Test loss: ', score[0])
        # print('Test accuracy:', score[1])