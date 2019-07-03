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
from sklearn.utils import class_weight

from Dataset import get_discriminator_train_data


class FRAGE():
    def __init__(self, dim, em_dim, max_words, maxlen, embedding_layer, class_num, isPairData, weight=0.1):

            uInput = Input(shape=(maxlen,))
            vInput = Input(shape=(maxlen,))
            disInput = Input(shape=(em_dim,))
            wordInput = Input(shape=(1,))

            self.encoder = Sequential()
            self.encoder.add(embedding_layer)

            bilstm = Bidirectional(LSTM(dim, return_sequences=True))
            pool = GlobalMaxPooling1D()

            # Generate Discriminator
            disDense1 = Dense(dim, activation="linear", use_bias=False, kernel_initializer=RandomUniform(-0.1, 0.1))
            disDense2 = Dense(1, activation="sigmoid")

            dout = disDense2(disDense1(disInput))

            self.discriminator = Model(disInput, dout)

            # disc_loss_mode = [0, -0.1, 0.1, 1, -1]
            self.discriminator.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'], loss_weights=[1])

            self.discriminator.trainable = False

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

            validity = self.discriminator(Flatten()(self.encoder(wordInput)))
            out = finalDense(dense(merge))
            self.model = Model([uInput, vInput] if isPairData else [uInput], out)

            self.model.compile(loss=loss,
                          optimizer='adam',
                          metrics=[metric])

            self.advModel = Model([uInput, vInput, wordInput] if isPairData else [uInput, wordInput], [out, validity])

            self.advModel.compile(loss=[loss, "binary_crossentropy"],
                             optimizer='adam',
                             loss_weights=[1, weight],
                             metrics=[metric, "acc"])


    def init(self, x_train, discMode, isPairData, batch_size, pop_percent=0.2):

        x, y = get_discriminator_train_data(x_train, discMode, isPairData)

        self.popular_x = x[:int(len(y) * pop_percent)]
        self.rare_x = x[int(len(y) * pop_percent):]
        self.popular_y = np.ones(batch_size)
        self.rare_y = np.zeros(batch_size)


    def train(self, x_train, y_train, epoch, batch_size, isPairData):
        t1 = time()
        for i in range(math.ceil(y_train.shape[0] / batch_size)):
            idx = np.random.randint(0, y_train.shape[0], batch_size)
            _x_train = x_train[idx] if not isPairData else [x_train[0][idx], x_train[1][idx]]
            _y_train = y_train[idx]

            idx = np.random.randint(0, len(self.popular_x), batch_size)
            _popular_x = self.popular_x[idx]
            idx = np.random.randint(0, len(self.rare_x), batch_size)
            _rare_x = self.rare_x[idx]

            _popular_x = self.encoder.predict(_popular_x).squeeze()
            _rare_x = self.encoder.predict(_rare_x).squeeze()

            d_loss_popular = self.discriminator.train_on_batch(_popular_x, self.popular_y)
            d_loss_rare = self.discriminator.train_on_batch(_rare_x, self.rare_y)

            d_loss = 0.5 * np.add(d_loss_popular, d_loss_rare)

            idx = np.random.randint(0, len(self.popular_x), int(batch_size / 2))
            _popular_x = self.popular_x[idx]
            idx = np.random.randint(0, len(self.rare_x), int(batch_size / 2))
            _rare_x = self.rare_x[idx]

            _popular_rare_x = np.concatenate([_popular_x, _rare_x])

            _popular_rare_y = np.concatenate([np.zeros(int(batch_size / 2)), np.ones(int(batch_size / 2))])

            # print(advModel.predict([_x_train, _popular_rare_x]))

            g_loss = self.advModel.train_on_batch(
                [_x_train, _popular_rare_x] if not isPairData else _x_train + [_popular_rare_x],
                [_y_train, _popular_rare_y])


        t2 = time()
        output = "%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %f] [%.1f s]" % (
            epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], t2 - t1)

        return output
