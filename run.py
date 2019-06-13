import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import argparse
from time import time
from Dataset import *
from LSTM import *
from datetime import datetime
from sklearn.utils import class_weight


def parse_args():
    parser = argparse.ArgumentParser(description="Run TFIDF-Word-Representation")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="adv_lstm")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="imdb")

    parser.add_argument('--d', type=int, default=128,
                        help='Dimension')

    parser.add_argument('--ml', type=int, default=10,
                        help='Maximum lenght of sequence')

    parser.add_argument('--mw', type=int, default=10000,
                        help='Maximum words')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Epoch number')

    parser.add_argument('--dm', type=str, default="tf",
                        help='Discriminator mode: tf or idf')

    parser.add_argument('--mode', type=int, default="2",
                        help='Mode: ')

    return parser.parse_args()


if __name__ == '__main__':
    start = time()

    args = parse_args()

    path = args.path
    dataset = args.data
    modelName = args.model
    dim = args.d
    max_words = args.mw
    maxlen = args.ml
    epochs = args.epochs
    discMode = args.dm
    modelMode = args.mode

    if dataset == "imdb":
        x_train, y_train, x_test, y_test = get_imbd(max_words, maxlen)

    print("Load model")
    runName = "%s_d%d_w%d_ml%d_%s_m%d_%s" % (
        modelName, dim, max_words, maxlen, discMode, modelMode, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    if modelName == "lstm":
        model = get_lstm(dim, max_words, maxlen)
    elif modelName == "adv_lstm":
        advModel, model, encoder, discriminator = get_adv_lstm(dim, max_words, maxlen, modelMode)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    if "adv" in modelName:
        disc_x, disc_y = get_discriminator_train_data(x_train, x_test, discMode)
        disc_class_weights = class_weight.compute_class_weight('balanced', np.unique(disc_y), disc_y)
        print(disc_class_weights)

    batch_size = 250
    for epoch in range(epochs):

        if "adv" in modelName:

            t1 = time()

            for i in range(int(x_train.shape[0] / batch_size)):
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                _x_train = x_train[idx]
                _y_train = y_train[idx]

                idx = np.random.randint(0, len(disc_x), batch_size)
                _disc_x = disc_x[idx]
                _disc_y = disc_y[idx]

                _disc_x = encoder.predict(_disc_x)

                loss = advModel.train_on_batch([_x_train, _disc_x], [_y_train, _disc_y],
                                               class_weight=[class_weights, disc_class_weights])

                adv_loss = discriminator.train_on_batch(_disc_x, _disc_y, class_weight=disc_class_weights)

            t2 = time()
            res = model.test_on_batch(x_test, y_test)
            dis_res = discriminator.test_on_batch(encoder.predict(disc_x), disc_y)
            # print(res[1], dis_res[1])
            t3 = time()

            output = "Epoch %d, train[%.1f s], Dloss: %f, Dacc: %f, Mloss: %f, acc: %f, test[%.1f s]" % (
                epoch, t2 - t1, adv_loss[0], adv_loss[1], loss[0], res[1], t3 - t2)
            with open(path + "out/%s.out" % runName, "a") as myfile:
                myfile.write(output + "\n")

            print(output)
        else:

            # TODO apply early stop here on validation set

            t1 = time()
            his = model.fit(x_train, y_train, batch_size=256, verbose=0, shuffle=True)
            t2 = time()
            res = model.test_on_batch(x_test, y_test)
            t3 = time()

            output = "Epoch %d, train[%.1f s], loss: %f, acc: %f, test[%.1f s]" % (
                epoch, t2 - t1, his.history['loss'][0], res[1], t3 - t2)

            with open(path+"out/%s.out" % runName, "a") as myfile:
                myfile.write(output+"\n")

            print(output)

    total_time = (time() - start) / 3600
    with open(path + "out/%s.out" % runName, "a") as myfile:
        myfile.write("Total time: %.2f h" % total_time + "\n")
