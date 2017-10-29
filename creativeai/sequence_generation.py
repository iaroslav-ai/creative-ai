"""
Generation of text, music, and other data sequential in nature.
"""

import numpy as np

# use sklearn for simplification of interfaces
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

# use keras for deep learning
from keras.models import Model
from keras.layers import Input, Dense, Activation, GRU, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy

from noxer.rnn import make_keras_picklable

make_keras_picklable()

class NextCharacterGenerator(BaseEstimator):
    """
    A model that learns to generate next character
    in a sequence.
    Such model has a form f: h -> R^n, where h is
    a representation of a history (eg previous text)
    and there are n possible next characters. A model
    outputs probabilities for every character being
    the next one.
    """
    def __init__(self, n_layers=1, n_neurons=128, batch_size=128, epochs=16):
        """

        Parameters
        ----------
        n_layers: int
            number of layers in the neural network

        n_neurons: int
            number of neurons in every layer
        """
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.epochs = epochs

        self.encoder = None
        self.keras_model = None

    def fit(self, X, y):
        """
        Fits a generative model based on input features
        X and output class y.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            Set of representation of a history

        y: array of shape [n_samples]
            Corresponding next characters
        """

        # encode all charaters as OneHot
        self.encoder = LabelBinarizer().fit(y)

        # transform labels
        y = self.encoder.transform(y)

        # define a model
        x = Input(shape=X.shape[1:])
        h = x

        for i in range(self.n_layers):
            h = GRU(self.n_neurons, return_sequences=True)(h)
            #h = LeakyReLU()(h)

        h = Flatten()(h)
        h = Dense(y.shape[-1])(h)

        model = Model(inputs=x, outputs=h)
        self.keras_model = model

        model.compile(SGD(lr=0.0001, momentum=0.1), categorical_crossentropy)

        # do the training
        model.fit(
            X, y, batch_size=self.batch_size, epochs=self.epochs
        )

        return self


    def predict_proba(self, X):
        """
        Estimate next characters for a set of inputs X.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            Set of representation of a history
        """

        # estimate probabilities for every history

        P = self.keras_model.predict(X)
        return P


    def predict(self, X):
        """
        Estimate next characters for a set of inputs X.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            Set of representation of a history
        """

        # estimate probabilities for every history

        P = self.predict_proba(X)
        y = self.encoder.inverse_transform(P)
        return y


    def score(self, X, y):
        y = self.encoder.transform(y)
        loss = log_loss(y, self.predict_proba(X))
        return 1.0 / (loss + 1e-7)


    def sample(self, X):
        """
        Sample next characters for a set of inputs X.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            Set of representation of a history
        """

        # estimate probabilities for every history

        P = self.keras_model.predict(X)

        y = []
        for p in P:
            character = np.random.choice(self.encoder.classes_, p)
            y.append(character)

        return y

