from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
import pandas as pd

def get_imbd(max_words=10000, maxlen=500):
    print('Loading data...')
    old = np.load
    np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

    np.load = old
    del (old)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, y_train, x_test, y_test



# MAX_SEQUENCE_LENGTH = 1000
# MAX_NUM_WORDS = 20000
# EMBEDDING_DIM = 300

def get_datasets(path, dataset, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH):

    if dataset == "QQP":

        df = pd.read_csv(path+"data/glue_data/QQP/train.tsv", sep="\t", error_bad_lines=False, nrows=1000)
        df_val = pd.read_csv(path+"data/glue_data/QQP/dev.tsv", sep="\t", error_bad_lines=False, nrows=1000)
        # there is no label on test set
        # df_test = pd.read_csv(path+"data/glue_data/QQP/test.tsv", sep="\t", error_bad_lines=False, nrows=1000)

        df.question1 = df.question1.astype(str)
        df.question2 = df.question2.astype(str)

        df_val.question1 = df_val.question1.astype(str)
        df_val.question2 = df_val.question2.astype(str)

        # df_test.question1 = df_test.question1.astype(str)
        # df_test.question2 = df_test.question2.astype(str)

        # corpus = df.question1.tolist() + df.question2.tolist() + df_val.question1.tolist() + df_val.question2.tolist() + df_test.question1.tolist() + df_test.question2.tolist()
        corpus = df.question1.tolist() + df.question2.tolist() + df_val.question1.tolist() + df_val.question2.tolist()

        # create the tokenizer
        t = Tokenizer()
        # fit the tokenizer on the documents
        t.fit_on_texts(corpus)

        # finally, vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(corpus)

        x1_train = tokenizer.texts_to_sequences(df.question1.tolist())
        x2_train = tokenizer.texts_to_sequences(df.question2.tolist())

        x1_val = tokenizer.texts_to_sequences(df_val.question1.tolist())
        x2_val = tokenizer.texts_to_sequences(df_val.question2.tolist())

        # x1_test = tokenizer.texts_to_sequences(df_test.question1.tolist())
        # x2_test = tokenizer.texts_to_sequences(df_test.question2.tolist())

        x1_train = pad_sequences(x1_train, maxlen=MAX_SEQUENCE_LENGTH)
        x2_train = pad_sequences(x2_train, maxlen=MAX_SEQUENCE_LENGTH)
        x_train = [x1_train, x2_train]

        x1_val = pad_sequences(x1_val, maxlen=MAX_SEQUENCE_LENGTH)
        x2_val = pad_sequences(x2_val, maxlen=MAX_SEQUENCE_LENGTH)
        x_val = [x1_val, x2_val]

        # x1_test = pad_sequences(x1_test, maxlen=MAX_SEQUENCE_LENGTH)
        # x2_test = pad_sequences(x2_test, maxlen=MAX_SEQUENCE_LENGTH)
        # x_test = [x1_test, x2_test]

        y_train = df.is_duplicate.values
        y_val = df_val.is_duplicate.values
        # print(df_test.columns)
        # y_test = df_test.is_duplicate.values

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        # return x_train, x_test, x_val, y_val, x_test, y_test, word_index
        return x_train, y_train, x_val, y_val, word_index



def get_discriminator_train_data(x_train, x_test, mode="tf", isPairData=False):

    # extract sentence-pair data or sentence data
    corpus = np.concatenate([x_train[0], x_train[1], x_test[0], x_test[1]]) if isPairData else np.concatenate([x_train, x_test])
    if mode == "tf":
        term_frequency = {}
        for i in corpus:
            for j in i:
                if j in term_frequency:
                    term_frequency[j] += 1
                else:
                    term_frequency[j] = 1

        term_frequency = {k: v for k, v in sorted(term_frequency.items(), key=lambda x: x[1])[::-1]}
        term_frequency = np.array(list(term_frequency.keys()))

        label = np.zeros(len(term_frequency))
        # set first 20% words as popular word
        label[:int(len(label) * 0.2)] = 1

        return term_frequency, label

    elif mode == "idf":

        corpus = []
        for i in np.concatenate([x_train, x_test]):
            corpus.append(' '.join(i.astype(str)))

        tfidf = TfidfVectorizer()
        tfidf.fit(corpus)

        wordIDF = {int(k): v for k, v in zip(tfidf.get_feature_names(), tfidf.idf_)}
        wordIDF = {k: v for k, v in sorted(wordIDF.items(), key=lambda x: x[1])[::-1]}
        wordIDF = np.array(list(wordIDF.keys()))


        label = np.zeros(len(wordIDF))
        # set first 20% words as popular word
        label[:int(len(label) * 0.2)] = 1

        return wordIDF, label


