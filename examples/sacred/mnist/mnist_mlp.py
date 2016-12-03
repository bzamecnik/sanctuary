"""
Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

A basic example from Keras.
Forked from: https://github.com/fchollet/keras/tree/master/examples

This is a basic example how to wrap the code as Sacred Experiment
(https://github.com/IDSIA/sacred).
"""

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sacred import Experiment, Ingredient
from tempfile import NamedTemporaryFile

dataset_ingredient = Ingredient('dataset')
net_ingredient = Ingredient('feed_forward_network')

ex = Experiment('mnist', ingredients=[dataset_ingredient, net_ingredient])

@dataset_ingredient.capture
def load_data(nb_classes=10):
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def prepare_X(X):
        X = X.reshape(X.shape[0], -1)
        X = X.astype('float32')
        X /= 255
        return X

    X_train, X_test = [prepare_X(X) for X in (X_train, X_test)]
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test

@net_ingredient.config
def net_ingredient_config():
    input_shape=(784,)
    nb_classes=10
    nb_layers=3
    layer_width=512
    dropout=0.2
    activation='relu'

@net_ingredient.capture
def create_model(input_shape, nb_classes, nb_layers, layer_width, dropout, activation):
    model = Sequential()
    for i in range(nb_layers - 1):
        if i == 0:
            model.add(Dense(layer_width, input_shape=input_shape))
        else:
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

@ex.automain
def main(batch_size, nb_epoch):
    X_train, Y_train, X_test, Y_test = load_data()
    model = create_model()

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))

    save_model(model)

    score = model.evaluate(X_test, Y_test, verbose=0)

    return {
        'history': history.history,
        'test_loss': score[0],
        'test_accuracy': score[1]
    }
