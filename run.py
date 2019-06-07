import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import argparse
from time import time
from Dataset import *
from LSTM import *
from datetime import datetime


def parse_args():

    parser = argparse.ArgumentParser(description="Run TFIDF-Word-Representation")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="adv_lstm")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="imdb")

    parser.add_argument('--d', type=int, default=128,
                        help='Dimension')

    parser.add_argument('--ml', type=int, default=500,
                        help='Maximum lenght of sequence')

    parser.add_argument('--mw', type=int, default=10000,
                        help='Maximum words')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Epoch number')

    parser.add_argument('--dm', type=str, default="tf",
                        help='Discriminator mode: tf or idf')

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



    if dataset == "imdb":

        x_train, y_train, x_test, y_test = get_imbd(max_words, maxlen)


    print("Load model")
    runName = "%s_d%d_w%d_ml%d_%s_%s" % (modelName, dim, max_words, maxlen, discMode, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    if modelName == "lstm":
        model = get_lstm(dim, max_words, maxlen)
    elif modelName == "adv_lstm":
        advModel, model, discriminator = get_adv_lstm(dim, max_words, maxlen)


    if "adv" in modelName:
        disc_x, disc_y = get_discriminator_train_data(x_train, x_test, discMode)


    for epoch in range(epochs):

        if "adv" in modelName:

            t1 = time()
            # sample discriminator data
            idx = np.random.randint(0, len(disc_x), x_train.shape[0])
            sample_disc_x  = disc_x[idx]
            sample_disc_y  = disc_y[idx]

            # Train the main model
            his = advModel.fit([x_train, sample_disc_x], [y_train, sample_disc_y], batch_size=256, verbose=0, shuffle=True)

            # Train the discriminator
            adv_loss = discriminator.train_on_batch(sample_disc_x, sample_disc_y)

            t2 = time()
            res = model.test_on_batch(x_test, y_test)
            t3 = time()

            output =  "Epoch %d, train[%.1f s], Dloss: %f, Dacc: %f, Mloss: %f, acc: %f, test[%.1f s]" % (epoch, t2-t1, adv_loss[0], adv_loss[1], his.history['loss'][0], res[1], t3-t2)


        else:

            t1 = time()
            his = model.fit(x_train, y_train, batch_size=256, verbose=0, shuffle=True)
            t2 = time()
            res = model.test_on_batch(x_test, y_test)
            t3 = time()

            output =  "Epoch %d, train[%.1f s], loss: %f, acc: %f, test[%.1f s]" % (epoch, t2-t1, his.history['loss'][0], res[1], t3-t2)


        with open(path+"out/%s.out" % runName, "a") as myfile:
            myfile.write(output+"\n")

        print(output)


    total_time = (time() - start ) / 3600
    with open(path + "out/%s.out" % runName, "a") as myfile:
        myfile.write("Total time: %.2f h" % total_time + "\n")