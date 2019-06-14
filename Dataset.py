from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from senteval.mrpc import MRPCEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.snli import SNLIEval
from senteval.sst import SSTEval
from senteval.sts import STSEval, STSBenchmarkEval
from senteval.trec import TRECEval
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.utils.np_utils import to_categorical


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

def get_datasets(path, dataset, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, isPairData):

    if dataset == "MRPC":

        # df = pd.read_csv(path + "data/MRPC/msr_paraphrase_train.txt", sep="\t", error_bad_lines=False, skiprows=1,
        #                  names=["label", "id1", "id2", "s1", "s2"])
        # df_test = pd.read_csv(path + "data/MRPC/msr_paraphrase_test.txt", sep="\t", error_bad_lines=False, skiprows=1,
        #                       names=["label", "id1", "id2", "s1", "s2"])

        mrpc = MRPCEval(path+"data/MRPC/")

        sen1_train = mrpc.sick_data['train']['X_A']
        sen2_train = mrpc.sick_data['train']['X_B']
        sen1_test = mrpc.sick_data['test']['X_A']
        sen2_test = mrpc.sick_data['test']['X_B']

        corpus = sen1_train + sen2_train + sen1_test + sen2_test

        y_train = mrpc.sick_data['train']["y"]
        y_test = mrpc.sick_data['test']["y"]

    elif dataset == "TREC":
        trec = TRECEval(path+"data/TREC/")

        x_train = trec.train["X"]
        y_train = trec.train["y"]
        x_test = trec.test["X"]
        y_test = trec.test["y"]

        corpus = x_train + x_test

    elif dataset == "SNLI":
        snli = SNLIEval(path+"data/SNLI/")
        corpus = snli.samples

        sen1_train = snli.data['train'][0]
        sen2_train = snli.data['train'][1]
        y_train = snli.data['train'][2]

        # sen1_val = snli.data['valid'][0]
        # sen2_val = snli.data['valid'][1]
        # y_val = snli.data['valid'][2]

        sen1_test = snli.data['test'][0]
        sen2_test = snli.data['test'][1]
        y_test = snli.data['test'][2]

    elif dataset in ["SICK_R", "SICK_E", "STS"]:
        if dataset == "SICK_R":
            sick = SICKRelatednessEval(path+"data/SICK/")
        elif dataset == "SICK_E":
            sick = STSBenchmarkEval(path + "data/STS/STSBenchmark/")
        elif dataset == "STS":
            sick = SICKEntailmentEval(path+"data/SICK/")

        sen1_train = sick.sick_data['train']['X_A']
        sen2_train = sick.sick_data['train']['X_B']

        sen1_val = sick.sick_data['dev']['X_A']
        sen2_val = sick.sick_data['dev']['X_B']

        sen1_test = sick.sick_data['test']['X_A']
        sen2_test = sick.sick_data['test']['X_B']

        corpus = sen1_train + sen2_train + sen1_val + sen2_val + sen1_test + sen2_test

        y_train = sick.sick_data['train']["y"]
        y_val = sick.sick_data['dev']["y"]
        y_test = sick.sick_data['test']["y"]

    elif dataset in ["SST2", "SST5"]:
        sst = SSTEval(path+"data/SST", nclasses=2 if dataset == "SST2" else 5)

        x_train = sst["train"]["X"]
        y_train = sst["train"]["y"]
        x_val = sst["dev"]["X"]
        y_val = sst["dev"]["y"]
        x_test = sst["test"]["X"]
        y_test = sst["test"]["y"]

        corpus = x_train + x_val + x_test


    elif dataset == "QQP":

        df = pd.read_csv(path + "data/glue_data/QQP/train.tsv", sep="\t",
                         names=["id", "qid1", "qid2", "s1", "s2", "label"], skiprows=1, error_bad_lines=False)
        df_test = pd.read_csv(path + "data/glue_data/QQP/dev.tsv", sep="\t",
                              names=["id", "qid1", "qid2", "s1", "s2", "label"], skiprows=1, error_bad_lines=False)

        df = df[~df.label.isna()]
        df_test = df_test[~df_test.label.isna()]

        df.s1 = df.s1.astype(str)
        df.s2 = df.s2.astype(str)

        df_test.s1 = df_test.s1.astype(str)
        df_test.s2 = df_test.s2.astype(str)

        y_train = df.label.values
        y_test = df_test.label.values

        sen1_train = df.s1.tolist()
        sen2_train = df.s2.tolist()
        sen1_test = df_test.s1.tolist()
        sen2_test = df_test.s2.tolist()

        corpus = sen1_train + sen2_train + sen1_test + sen2_test

    # create the tokenizer
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(corpus)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    if isPairData:

        x1_train = tokenizer.texts_to_sequences(sen1_train)
        x2_train = tokenizer.texts_to_sequences(sen2_train)

        x1_test = tokenizer.texts_to_sequences(sen1_test)
        x2_test = tokenizer.texts_to_sequences(sen2_test)

        x1_train = pad_sequences(x1_train, maxlen=MAX_SEQUENCE_LENGTH)
        x2_train = pad_sequences(x2_train, maxlen=MAX_SEQUENCE_LENGTH)
        x_train = [x1_train, x2_train]

        x1_val = pad_sequences(x1_test, maxlen=MAX_SEQUENCE_LENGTH)
        x2_val = pad_sequences(x2_test, maxlen=MAX_SEQUENCE_LENGTH)
        x_test = [x1_val, x2_val]

    else:
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

    return x_train, y_train, x_test, y_test, word_index


def get_discriminator_train_data(x_train, x_test, mode="tf", isPairData=False):
    # extract sentence-pair data or sentence data
    corpus = np.concatenate([x_train[0], x_train[1], x_test[0], x_test[1]]) if isPairData else np.concatenate(
        [x_train, x_test])
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
