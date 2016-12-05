"""
Trains a simple deep NN on the MNIST dataset.

Allows to use multiple models (basic fully conncted layers, convolution layers).

Based on some basic example from Keras. Forked from:
https://github.com/fchollet/keras/tree/master/examples

This is a basic example how to wrap the code as Sacred Experiment
(https://github.com/IDSIA/sacred).

How to use:

# help
$ python mnist.py --help

# normal run
$ python mnist.py

# print the configuration values
$ python mnist.py print_config

# change some parameters
$ python mnist.py with nb_epoch=10 model_fc.nb_layers=4 model_fc.dropout=0.5

# change the architecture
$ python mnist.py with model_arch='conv' model_conv.nb_filters=30
"""

from keras.datasets import mnist
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sacred import Experiment, Ingredient
from sacred.utils import apply_backspaces_and_linefeeds
from tempfile import NamedTemporaryFile

from keras_sacred import TrainingHistoryToSacredInfo

dataset_ingredient = Ingredient('dataset')
net_fc_ingredient = Ingredient('model_fc')
net_conv_ingredient = Ingredient('model_conv')

ex = Experiment('mnist', ingredients=[
    dataset_ingredient,
    net_fc_ingredient,
    net_conv_ingredient
])
# ignore the intermediate progressbar characters
ex.captured_out_filter = apply_backspaces_and_linefeeds

@dataset_ingredient.capture
def load_data(nb_classes=10):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def prepare_X(X):
        return X.astype('float32') / 255

    X_train, X_test = [prepare_X(X) for X in (X_train, X_test)]
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test

@net_fc_ingredient.config
def net_fc_ingredient_config():
    input_shape=(28, 28)
    nb_classes=10
    nb_layers=3
    layer_width=512
    dropout=0.2
    activation='relu'

@net_fc_ingredient.capture
def create_fc_model(input_shape, nb_classes, nb_layers, layer_width, dropout, activation):
    model = Sequential()
    model.add(Reshape((input_shape[0] * input_shape[1],), input_shape=input_shape))
    for i in range(nb_layers - 1):
        model.add(Dense(layer_width))
        model.add(Activation(activation))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

@net_conv_ingredient.config
def net_conv_ingredient_config():
    input_shape=(28, 28)
    nb_classes=10
    nb_conv_blocks=1
    dropout=0.2
    activation='relu'
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    fc_layer_width = 128

@net_conv_ingredient.capture
def create_conv_model(input_shape, nb_classes, nb_conv_blocks,
        nb_filters, pool_size, kernel_size, dropout, activation,
        fc_layer_width):
    model = Sequential()
    nb_rows, nb_cols = input_shape
    # add one dimension for the convolution filters
    model.add(Reshape((nb_rows, nb_cols, 1), input_shape=input_shape))
    for i in range(nb_conv_blocks):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid'))
        model.add(Activation(activation))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=pool_size))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(fc_layer_width))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

@ex.capture
def create_model(model_arch):
    if model_arch == 'fc':
        return create_fc_model()
    elif model_arch == 'conv':
        return create_conv_model()

def save_model(model):
    """
    Saved the model as an Sacred artifact.
    """
    with NamedTemporaryFile(suffix='_model.h5') as model_file:
        model.save(model_file.name)
        ex.add_artifact(model_file.name)

@ex.config
def ex_config():
    batch_size = 128
    nb_epoch = 20
    model_arch = 'fc'

@ex.automain
def main(_run, batch_size, nb_epoch, model_arch):
    X_train, Y_train, X_test, Y_test = load_data()
    model = create_model()

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test),
                        callbacks=[TrainingHistoryToSacredInfo(_run)])

    save_model(model)

    score = model.evaluate(X_test, Y_test, verbose=0)

    return {
        'history': history.history,
        'test_loss': score[0],
        'test_accuracy': score[1]
    }
