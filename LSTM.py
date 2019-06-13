from keras.models import Sequential, Model
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Input, Flatten





def get_lstm(dim, max_words, maxlen):

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

def get_adv_lstm(dim, max_word, maxlen, mode):

    wordInput = Input(shape=(1,))
    disInput = Input(shape=(dim,))
    seqInput = Input(shape=(maxlen,))

    emb = Embedding(max_word, dim)

    wordEmb = Flatten()(emb(wordInput))


    encoder = Model(wordInput, wordEmb)


    seqEmb = emb(seqInput)

    disDense1 = Dense(int(dim * 1.5), activation="relu")
    disDense2 = Dense(1, activation="sigmoid")

    dout = disDense2(disDense1(disInput))

    discriminator = Model(disInput, dout)


    disc_loss_mode = [0, -0.1, 0.1, 1]
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'], loss_weights=[disc_loss_mode[mode]])

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