from keras.callbacks import CSVLogger, Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, GlobalMaxPooling1D, Lambda
from keras.layers import LSTM, Input, Multiply, Concatenate
from keras import backend as K
from time import time

# InferSent, Facebook https://arxiv.org/pdf/1705.02364.pdf
class BiLSTM():
    def __init__(self, dim, max_words, maxlen, embedding_layer, class_num, isPairData):

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
        self.model = Model([uInput, vInput] if isPairData else [uInput], out)

        self.model.compile(loss=loss,
                           optimizer='adam',
                           metrics=[metric])

    def init(self, path, runName):
        self.path = path
        self.runName = runName

    def train(self, x_train, y_train, epoch, batch_size, isPairData):

        # Callbacks
        # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='min')
        # cp = ModelCheckpoint(path + 'h5/%s.h5' % runName, monitor='val_loss', verbose=2, save_best_only=True,
        #                      save_weights_only=False,
        #                      mode='min', period=1)
        # logger = CSVLogger(self.path + "out/%s.out" % self.runName)


        # class Eval(Callback):
        #
        #     def on_epoch_end(self, epoch, logs=None):
        #         def on_batch_end(self, batch, logs=None):
        #
        # def on_epoch_end(self, epoch, logs=None):
        # val_res = model.test_on_batch(x_val, y_val)
        # test_res = model.test_on_batch(x_test, y_test)
        # output = "Val acc: %f, Test acc: %f" % (val_res[1], test_res[1])
        # save2file(self.path + "out/%s.res" % self.runName, output)
        #
        #
        # his = self.model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=1, shuffle=True,
        #                 validation_data=(x_val, y_val), callbacks=[Eval(), logger])
        t1 = time()
        his = self.model.fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=1, shuffle=True)
        t2 = time()
        output = "%d loss: %f, acc: %f [%.1f s]" % (
            epoch, his.history["loss"][0], his.history["acc"][0], t2 - t1)
        return output
