from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np


def get_imbd(max_words=10000, maxlen=500):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, y_train, x_test, y_test


def get_discriminator_train_data(x_train, x_test):
    term_frequency = {}
    for i in np.concatenate([x_train, x_test]):
        for j in i:
            if j in term_frequency:
                term_frequency[j] += 1
            else:
                term_frequency[j] = 1

    term_frequency = {k: v for k, v in sorted(term_frequency.items(), key=lambda x: x[1])[::-1]}
    term_frequency = np.array(list(term_frequency.keys()))

    label = np.zeros(len(term_frequency))
    # set first 20% words as popular word
    label[:int(term_frequency * 0.2)] = 1

    return term_frequency, label


