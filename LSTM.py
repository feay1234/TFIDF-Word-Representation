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

def get_adv_lstm(dim, max_word, maxlen):

    wordInput = Input(shape=(1,))
    seqInput = Input(shape=(maxlen,))

    emb = Embedding(max_word, dim)

    wordEmb = Flatten()(emb(wordInput))
    seqEmb = emb(seqInput)

    disDense1 = Dense(int(dim * 1.5), activation="relu")
    disDense2 = Dense(1, activation="sigmoid")

    dout = disDense2(disDense1(wordEmb))

    discriminator = Model(wordInput, dout)

    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

    lstm = LSTM(dim, dropout=0.2, recurrent_dropout=0.2)

    mDense = Dense(1, activation='sigmoid')

    mout = mDense(lstm(seqEmb))

    model = Model(seqInput, mout)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    discriminator.trainable = False

    advModel = Model([seqInput, wordInput], [mout, dout])

    advModel.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                  optimizer='adam',
                  loss_weights=[0.9, 0.1],
                  metrics=['accuracy'])


    return advModel, model, discriminator
