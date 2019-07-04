from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Bidirectional, GlobalMaxPooling1D, Lambda
from keras.layers import LSTM, Input, Flatten, Subtract, Multiply, Concatenate
from keras.initializers import RandomUniform
import tensorflow as tf
from time import time
from keras import backend as K
from keras.constraints import MinMaxNorm
import numpy as np
import math
# Keras version
# https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/aae.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight



class FRAGE():


    def __init__(self, dim, em_dim, max_words, maxlen, embedding_layer, class_num, isPairData, weight=0.1, weight2=0.1, mode=0):

        self.mode = mode
        self.dim = dim
        self.em_dim = em_dim

        uInput = Input(shape=(maxlen,))
        vInput = Input(shape=(maxlen,))
        wordInput = Input(shape=(1,))
        wordRegInput = Input(shape=(1,))

        self.encoder = Sequential()
        self.encoder.add(embedding_layer)

        bilstm = Bidirectional(LSTM(dim, return_sequences=True))
        pool = GlobalMaxPooling1D()

        if self.mode < 2:
            self.discriminator = self.generate_binary_discriminator() if mode == 0 else self.generate_reg_discriminator()
            self.discriminator.trainable = False
        elif self.mode == 2:
            self.discriminator_b = self.generate_binary_discriminator()
            self.discriminator_r = self.generate_reg_discriminator()
            self.discriminator_b.trainable = False
            self.discriminator_r.trainable = False


        uEmb = pool(bilstm(self.encoder(uInput)))

        if not isPairData:
            merge = uEmb
        else:

            vEmb = pool(bilstm(self.encoder(vInput)))

            def abs_diff(X):
                return K.abs(X[0] - X[1])

            abs = Lambda(abs_diff)

            concat = Concatenate()([uEmb, vEmb])
            sub = abs([uEmb, vEmb])
            mul = Multiply()([uEmb, vEmb])

            merge = Concatenate()([concat, sub, mul])

        dense = Dense(dim, activation="relu")

        if class_num == 1:
            finalDense = Dense(1, activation="linear", name="finalDense")
            loss = "mean_squared_error"
            metric = "mse"
        elif class_num == 2:
            finalDense = Dense(1, activation="sigmoid", name="finalDense")
            loss = "binary_crossentropy"
            metric = "acc"
        else:
            finalDense = Dense(class_num, activation="softmax", name="finalDense")
            loss = "categorical_crossentropy"
            metric = "acc"

        if self.mode < 2:
            validity = self.discriminator(Flatten()(self.encoder(wordInput)))
        else:
            validity_b = self.discriminator_b(Flatten()(self.encoder(wordInput)))
            validity_r = self.discriminator_r(Flatten()(self.encoder(wordRegInput)))

        out = finalDense(dense(merge))
        self.model = Model([uInput, vInput] if isPairData else [uInput], out)

        self.model.compile(loss=loss,
                           optimizer='adam',
                           metrics=[metric])

        if self.mode < 2:
            self.advModel = Model([uInput, vInput, wordInput] if isPairData else [uInput, wordInput], [out, validity])
            self.advModel.compile(loss=[loss, "binary_crossentropy" if self.mode == 0 else "mean_squared_error"],
                                  optimizer='adam',
                                  loss_weights=[1, weight],
                                  metrics=[metric, "acc" if self.mode == 0 else "mse"])
        elif self.mode == 2:
            self.advModel = Model([uInput, vInput, wordInput, wordRegInput] if isPairData else [uInput, wordInput, wordRegInput],
                                  [out, validity_b, validity_r])
            self.advModel.compile(loss=[loss, "binary_crossentropy", "mean_squared_error"],
                                  optimizer='adam',
                                  loss_weights=[1, weight, weight2],
                                  metrics=[metric, "acc", "mse"])

    def init(self, x_train, discMode, isPairData, pop_percent=0.2):

        x, y = self.get_discriminator_train_data(x_train, discMode, isPairData)

        if self.mode == 0:
            self.popular_x = x[:int(len(y) * pop_percent)]
            self.rare_x = x[int(len(y) * pop_percent):]
        elif self.mode == 1:
            self.disc_x = x
            self.disc_y = y
        elif self.mode == 2:
            self.popular_x = x[:int(len(y) * pop_percent)]
            self.rare_x = x[int(len(y) * pop_percent):]
            self.disc_x = x
            self.disc_y = y

    def generate_binary_discriminator(self):
        disInput = Input(shape=(self.em_dim,))

        # Generate Discriminator
        disDense1 = Dense(self.dim, activation="linear", use_bias=False, kernel_initializer=RandomUniform(-0.1, 0.1))
        disDense2 = Dense(1, activation="sigmoid")
        dout = disDense2(disDense1(disInput))
        model = Model(disInput, dout)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'], loss_weights=[1])

        return model

    def generate_reg_discriminator(self):
        disInput = Input(shape=(self.em_dim,))

        # Generate Discriminator
        disDense1 = Dense(self.dim, activation="linear", use_bias=False, kernel_initializer=RandomUniform(-0.1, 0.1))
        disDense2 = Dense(1, activation="linear")
        dout = disDense2(disDense1(disInput))
        model = Model(disInput, dout)

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'], loss_weights=[1])

        return model


    def train(self, x_train, y_train, epoch, batch_size, isPairData):
        t1 = time()
        for i in range(math.ceil(y_train.shape[0] / batch_size)):
            idx = np.random.randint(0, y_train.shape[0], batch_size)
            _x_train = x_train[idx] if not isPairData else [x_train[0][idx], x_train[1][idx]]
            _y_train = y_train[idx]


            if self.mode == 0:
                idx = np.random.randint(0, len(self.popular_x), batch_size)
                _popular_x = self.popular_x[idx]
                idx = np.random.randint(0, len(self.rare_x), batch_size)
                _rare_x = self.rare_x[idx]

                _popular_x = self.encoder.predict(_popular_x).squeeze()
                _rare_x = self.encoder.predict(_rare_x).squeeze()
                d_loss_popular = self.discriminator.train_on_batch(_popular_x, np.ones(len(_popular_x)))
                d_loss_rare = self.discriminator.train_on_batch(_rare_x, np.zeros(len(_rare_x)))
                d_loss = 0.5 * np.add(d_loss_popular, d_loss_rare)

                idx = np.random.randint(0, len(self.popular_x), int(batch_size / 2))
                _popular_x = self.popular_x[idx]
                idx = np.random.randint(0, len(self.rare_x), int(batch_size / 2))
                _rare_x = self.rare_x[idx]

                _popular_rare_x = np.concatenate([_popular_x, _rare_x])

                _popular_rare_y = np.concatenate([np.zeros(int(batch_size / 2)), np.ones(int(batch_size / 2))])

                g_loss = self.advModel.train_on_batch(
                    [_x_train, _popular_rare_x] if not isPairData else _x_train + [_popular_rare_x],
                    [_y_train, _popular_rare_y])

            elif self.mode == 1:

                idx = np.random.randint(0, len(self.disc_x), batch_size)
                _disc_x = self.disc_x[idx]
                _disc_y = self.disc_y[idx]
                _disc_x = self.encoder.predict(_disc_x).squeeze()
                d_loss = self.discriminator.train_on_batch(_disc_x, _disc_y)

                idx = np.random.randint(0, len(self.disc_x), batch_size)
                _disc_x = self.disc_x[idx]
                _disc_y = self.disc_y[idx]

                g_loss = self.advModel.train_on_batch(
                    [_x_train, _disc_x] if not isPairData else _x_train + [_disc_x],
                    [_y_train, _disc_y])

            elif self.mode == 2:

                idx = np.random.randint(0, len(self.popular_x), batch_size)
                _popular_x = self.popular_x[idx]
                idx = np.random.randint(0, len(self.rare_x), batch_size)
                _rare_x = self.rare_x[idx]

                _popular_x = self.encoder.predict(_popular_x).squeeze()
                _rare_x = self.encoder.predict(_rare_x).squeeze()
                d_loss_popular = self.discriminator_b.train_on_batch(_popular_x, np.ones(len(_popular_x)))
                d_loss_rare = self.discriminator_b.train_on_batch(_rare_x, np.zeros(len(_rare_x)))

                idx = np.random.randint(0, len(self.disc_x), batch_size)
                _disc_x = self.disc_x[idx]
                _disc_y = self.disc_y[idx]
                _disc_x = self.encoder.predict(_disc_x).squeeze()
                d_loss = self.discriminator_r.train_on_batch(_disc_x, _disc_y)

                d_loss = 0.5 * np.add(np.add(d_loss_popular, d_loss_rare), d_loss)

                idx = np.random.randint(0, len(self.disc_x), batch_size)
                _disc_x = self.disc_x[idx]
                _disc_y = self.disc_y[idx]


                idx = np.random.randint(0, len(self.popular_x), int(batch_size / 2))
                _popular_x = self.popular_x[idx]
                idx = np.random.randint(0, len(self.rare_x), int(batch_size / 2))
                _rare_x = self.rare_x[idx]

                _popular_rare_x = np.concatenate([_popular_x, _rare_x])

                _popular_rare_y = np.concatenate([np.zeros(int(batch_size / 2)), np.ones(int(batch_size / 2))])

                g_loss = self.advModel.train_on_batch(
                    [_x_train, _popular_rare_x, _disc_x] if not isPairData else _x_train + [_popular_rare_x, _disc_x],
                    [_y_train, _popular_rare_y, _disc_y])


        t2 = time()
        output = "%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %f] [%.1f s]" % (
            epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], t2 - t1)

        return output

    # def get_discriminator_train_data(x_train, x_test, mode="tf", isPairData=False):
    def get_discriminator_train_data(self, x_train, mode="tf", isPairData=False, isBinary=True):
        # extract sentence-pair data or sentence data
        corpus = np.concatenate([x_train[0], x_train[1]]) if isPairData else x_train
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

            label = np.zeros(len(term_frequency)) if isBinary else np.array(list(term_frequency.values()))

            return term_frequency, label

        elif mode == "idf":

            corpus = []
            tmp = np.concatenate([x_train[0], x_train[1]]) if isPairData else x_train
            for i in tmp:
                corpus.append(' '.join(i.astype(str)))

            tfidf = TfidfVectorizer()
            tfidf.fit(corpus)

            wordIDF = {int(k): v for k, v in zip(tfidf.get_feature_names(), tfidf.idf_)}
            wordIDF = {k: v for k, v in sorted(wordIDF.items(), key=lambda x: x[1])[::-1]}
            wordIDF = np.array(list(wordIDF.keys()))

            label = np.zeros(len(wordIDF)) if isBinary else np.array(list(wordIDF.values()))
            # set first 20% words as popular word
            # label[:int(len(label) * 0.2)] = 1

            return wordIDF, label
