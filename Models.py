from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Bidirectional, GlobalMaxPooling1D, Lambda
from keras.layers import LSTM, Input, Flatten, Subtract, Multiply, Concatenate
from keras.initializers import RandomUniform
import tensorflow as tf
from keras import backend as K
from keras.constraints import MinMaxNorm

def get_lstm(dim, max_words, maxlen, class_num):
    seqInput = Input(shape=(maxlen,))
    emb = Embedding(max_words, dim)
    seqEmb = emb(seqInput)
    lstm = LSTM(dim, dropout=0.2, recurrent_dropout=0.2)
    mDense = Dense(1, activation='sigmoid')

    mout = mDense(lstm(seqEmb))
    model = Model(seqInput, mout)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# InferSent, Facebook https://arxiv.org/pdf/1705.02364.pdf
def get_bilstm_maxpool(dim, max_words, maxlen, embedding_layer, class_num, isPairData):
    uInput = Input(shape=(maxlen,))
    vInput = Input(shape=(maxlen,))

    encoder = Sequential()
    encoder.add(embedding_layer)
    encoder.add(Bidirectional(LSTM(dim, return_sequences=True)))
    encoder.add(GlobalMaxPooling1D())

    uEmb = encoder(uInput)
    if not isPairData:
        merge = uEmb
    else:

        vEmb = encoder(vInput)

        def abs_diff(X):
            return K.abs(X[0] - X[1])

        abs = Lambda(abs_diff)

        concat = Concatenate()([uEmb, vEmb])
        sub = abs([uEmb, vEmb])
        mul = Multiply()([uEmb, vEmb])

        merge = Concatenate()([concat, sub, mul])

    dense = Dense(dim, activation="relu")

    if class_num == 1:
        finalDense = Dense(1, activation="linear")
        loss = "mean_squared_error"
        metric = "mse"
    elif class_num == 2:
        finalDense = Dense(1, activation="sigmoid")
        loss = "binary_crossentropy"
        metric = "acc"
    else:
        finalDense = Dense(class_num, activation="softmax")
        loss = "categorical_crossentropy"
        metric = "acc"

    out = finalDense(dense(merge))
    model = Model([uInput, vInput] if isPairData else [uInput], out)

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=[metric])

    return model


def get_adv_bilstm_maxpool(dim, em_dim, max_words, maxlen, embedding_layer, class_num, isPairData):
    uInput = Input(shape=(maxlen,))
    vInput = Input(shape=(maxlen,))
    disInput = Input(shape=(em_dim,))

    encoder = Sequential()
    encoder.add(embedding_layer)

    bilstm = Bidirectional(LSTM(dim, return_sequences=True))
    pool = GlobalMaxPooling1D()

    # Generate Discriminator
    disDense1 = Dense(dim, activation="linear", use_bias=False, kernel_initializer=RandomUniform(-0.1,0.1))
    disDense2 = Dense(1, activation="sigmoid")

    dout = disDense2(disDense1(disInput))

    discriminator = Model(disInput, dout)

    # disc_loss_mode = [0, -0.1, 0.1, 1, -1]
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'], loss_weights=[1])

    discriminator.trainable = False

    uEmb = pool(bilstm(encoder(uInput)))
    if not isPairData:
        merge = uEmb
    else:

        vEmb = pool(bilstm(encoder(vInput)))

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

    out = finalDense(dense(merge))
    model = Model([uInput, vInput] if isPairData else [uInput], out)

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=[metric])

    advModel = Model([uInput, vInput, disInput] if isPairData else [uInput, disInput], [out, dout])

    advModel.compile(loss=[loss, "binary_crossentropy"],
                     optimizer='adam',
                     loss_weights=[1, -0.1],
                     metrics=[metric, "acc"])

    return advModel, model, encoder, discriminator

# Keras version
# https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/aae.py
def get_adv_bilstm_maxpool_keras(dim, em_dim, max_words, maxlen, embedding_layer, class_num, isPairData):
    uInput = Input(shape=(maxlen,))
    vInput = Input(shape=(maxlen,))
    disInput = Input(shape=(em_dim,))
    wordInput = Input(shape=(1,))

    encoder = Sequential()
    encoder.add(embedding_layer)

    bilstm = Bidirectional(LSTM(dim, return_sequences=True))
    pool = GlobalMaxPooling1D()

    # Generate Discriminator
    disDense1 = Dense(dim, activation="linear", use_bias=False, kernel_initializer=RandomUniform(-0.1,0.1))
    disDense2 = Dense(1, activation="sigmoid")

    dout = disDense2(disDense1(disInput))

    discriminator = Model(disInput, dout)

    # disc_loss_mode = [0, -0.1, 0.1, 1, -1]
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'], loss_weights=[1])

    discriminator.trainable = False

    uEmb = pool(bilstm(encoder(uInput)))
    if not isPairData:
        merge = uEmb
    else:

        vEmb = pool(bilstm(encoder(vInput)))

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

    validity = discriminator(Flatten()(encoder(wordInput)))
    out = finalDense(dense(merge))
    model = Model([uInput, vInput] if isPairData else [uInput], out)

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=[metric])

    advModel = Model([uInput, vInput, wordInput] if isPairData else [uInput, wordInput], [out, validity])

    advModel.compile(loss=[loss, "binary_crossentropy"],
                     optimizer='adam',
                     loss_weights=[0.999, 0.001],
                     metrics=[metric, "acc"])

    return advModel, model, encoder, discriminator


def get_adv_lstm(dim, max_word, maxlen, mode):
    wordInput = Input(shape=(1,))
    disInput = Input(shape=(dim,))
    seqInput = Input(shape=(maxlen,))

    emb = Embedding(max_word, dim)

    wordEmb = Flatten()(emb(wordInput))

    encoder = Model(wordInput, wordEmb)

    seqEmb = emb(seqInput)

    lstm = LSTM(dim, dropout=0.2, recurrent_dropout=0.2)

    mDense = Dense(1, activation='sigmoid')

    mout = mDense(lstm(seqEmb))

    model = Model(seqInput, mout)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    discriminator.trainable = False

    advModel = Model([seqInput, disInput], [mout, dout])

    advModel.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                     optimizer='adam',
                     loss_weights=[1, -0.1],
                     metrics=['accuracy'])

    return advModel, model, encoder, discriminator

    # advModel, model, encoder, discriminator = get_adv_lstm(2, 10, 3)
    # import numpy as np
    # print(encoder.predict(np.arange(10)).shape)
    # print(model.predict(np.random.randint(10, size=(10,3))).shape)
    #
    # x_test = np.random.randint(10, size=(10,3))
    # y_test = np.random.randint(2, size=(10))
    # print(model.test_on_batch(x_test, y_test))
