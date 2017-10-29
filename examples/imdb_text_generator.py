"""
Learn to generate text similar to IMDB dataset feedbacks!
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_union, make_pipeline

from noxer.sequences import SequenceEstimator, SequenceTransformer, \
    Subsequensor, Seq1Dto2D, FlattenShape, PadSubsequence
from noxer.preprocessing import OneHotEncoder

from creativeai.datasets import read_imdb
from creativeai.sequence_generation import NextCharacterGenerator

data = read_imdb()

X = data['review'].as_matrix()
X = X[:10]

# outputs: next character
y = np.copy(X)

# add initial '_' symbol
X = np.array(["_"+v[:-1] for v in X])

# explicit output labels
y = np.array([list(st) for st in y])

feats = make_pipeline(
    Seq1Dto2D(), # "abc" -> [['a'], ['b'], ['c']]
    SequenceTransformer(
        make_pipeline(
            OneHotEncoder(), # encode characters
            StandardScaler() # scale values
        )
    ),
    PadSubsequence(length=20),
    #FlattenShape()
)

estimator = make_pipeline(
    feats,
    NextCharacterGenerator(epochs=4, n_neurons=256, batch_size=256)
)

# generate subsequences of text
trainer = SequenceEstimator(
    estimator=estimator,
    max_subsequence=100
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
trainer.fit(X_train, y_train)
print(trainer.score(X_test, y_test))

import pickle as pc
pc.dump(trainer, open('model.bin', 'wb'))