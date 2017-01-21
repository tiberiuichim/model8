""" Utilities dealing with training models
"""

# from joblib import Memory

from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

np.random.seed(1234)


def make_tokenizer(texts, max_nb_words=-1):
    """ Builds and fits the tokenizer
    """

    if max_nb_words == -1:
        tokenizer = Tokenizer()
    else:
        tokenizer = Tokenizer(nb_words=max_nb_words)

    import pdb; pdb.set_trace()
    tokenizer.fit_on_texts(texts)

    return tokenizer


def split_data(tokenizer, sentences, labels, max_seq_len=-1, vsplit=0.2):
    """ Transforms the list of sentences to training and testing sequences
    """

    sequences = tokenizer.texts_to_sequences(sentences)
    # TODO: autocalculate maxlen
    data = pad_sequences(sequences, maxlen=max_seq_len)

    labels = np.asarray(labels)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]    # cool way to shufle two arrays in same order
    nb_validation_samples = int(vsplit * data.shape[0])

    X_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]

    X_test = data[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]

    return (X_train, y_train, X_test, y_test, tokenizer)


def make_model(train_val, train_labels, test_val, test_labels, tokenizer,
               max_nb_words):

    print("Train shape", train_val.shape)
    print("Test shape", test_val.shape)

    print('Training model.')

    model = Sequential()
    model.add(Embedding(max_nb_words, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print("Compiling model")
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Fiting model")
    model.fit(train_val, train_labels, validation_data=(test_val, test_labels),
              nb_epoch=3, batch_size=128)

    return model
