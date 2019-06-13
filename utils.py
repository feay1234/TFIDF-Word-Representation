import numpy as np
import os

from keras.initializers import Constant
from keras.layers import Embedding

BASE_DIR = 'data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'w2v')

def get_pretrain_embeddings(MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, word_index):
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
            break

    print('Found %s word vectors.' % len(embeddings_index))

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if embedding_vector.shape[0] == 0:
                continue
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH)

    return embedding_layer