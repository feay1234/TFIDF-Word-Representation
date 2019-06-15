import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import argparse
from time import time
from Dataset import *
from Models import *
from datetime import datetime
from sklearn.utils import class_weight
import math
from utils import get_pretrain_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Run TFIDF-Word-Representation")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="adv_bilstm2")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="TREC")

    parser.add_argument('--d', type=int, default=300,
                        help='Dimension')
    parser.add_argument('--ed', type=int, default=300,
                        help='Embedding Dimension')

    parser.add_argument('--ml', type=int, default=100,
                        help='Maximum lenght of sequence')

    parser.add_argument('--mw', type=int, default=10000,
                        help='Maximum words')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Epoch number')

    parser.add_argument('--dm', type=str, default="tf",
                        help='Discriminator mode: tf or idf')

    parser.add_argument('--mode', type=int, default="1",
                        help='Mode:')

    parser.add_argument('--bs', type=int, default=32,
                        help='Batch Size:')
    parser.add_argument('--w', type=float, default=0.1,
                        help='Weight:')

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
    emb_dim = args.ed
    batch_size = args.bs
    weight = args.w

    # epochs = 1


    isPairData = True if dataset in ["QQP", "MRPC", "SICK_R", "SICK_E", "SNLI", "STS"] else False

    x_train, y_train, x_val, y_val, x_test, y_test, word_index, class_num, maxlen = get_datasets(path, dataset,
                                                                                                 max_words,
                                                                                                 maxlen, isPairData)
    embedding_layer = get_pretrain_embeddings(path, max_words, emb_dim, maxlen, word_index)

    runName = "%s_%s_d%d_w%d_ml%d_%s_m%d_%s" % (dataset,
                                                modelName, dim, max_words, maxlen, discMode, modelMode,
                                                datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    print("Load model: %s" % runName)

    if modelName == "lstm":
        model = get_lstm(dim, max_words, maxlen, class_num)
    elif modelName == "bilstm":
        model = get_bilstm_maxpool(dim, max_words, maxlen, embedding_layer, class_num, isPairData)
    elif modelName == "adv_bilstm":
        advModel, model, encoder, discriminator = get_adv_bilstm_maxpool(dim, emb_dim, max_words, maxlen,
                                                                         embedding_layer, class_num, isPairData)
    elif modelName == "adv_bilstm2":
        advModel, model, encoder, discriminator = get_adv_bilstm_maxpool_keras(dim, emb_dim, max_words, maxlen,
                                                                         embedding_layer, class_num, isPairData, weight)

    if "adv" in modelName:
        disc_x, disc_y = get_discriminator_train_data(x_train, x_test, discMode, isPairData)

        # set first 20% words as popular word
        if modelName == "adv_bilstm":
            disc_y[:int(len(disc_y) * 0.2)] = 1
            disc_class_weights = class_weight.compute_class_weight('balanced', np.unique(disc_y), disc_y)
            print("Disc_class_weight: ", disc_class_weights)
        else:
            # disc_y[:int(len(disc_y) * 0.5)] = 1
            popular_x = disc_x[:int(len(disc_y) * 0.2)]
            rare_x = disc_x[int(len(disc_y) * 0.2):]
            popular_y = np.ones(batch_size)
            rare_y = np.zeros(batch_size)

    if "adv" in modelName:
        minLoss = 9999999
        for epoch in range(epochs):

            t1 = time()

            if modelMode == "adv_bilstm":

                for i in range(math.ceil(y_train.shape[0] / batch_size)):


                    idx = np.random.randint(0, y_train.shape[0], batch_size)
                    _x_train = x_train[idx] if not isPairData else [x_train[0][idx], x_train[1][idx]]
                    _y_train = y_train[idx]

                    idx = np.random.randint(0, len(disc_x), batch_size)
                    _disc_x = disc_x[idx]
                    _disc_y = disc_y[idx]

                    _disc_x = encoder.predict(_disc_x).squeeze()
                    #
                    if modelMode == 1:
                        dis_loss = discriminator.train_on_batch(_disc_x, _disc_y, class_weight=disc_class_weights)
                    elif modelMode == 2:
                        dis_loss = discriminator.train_on_batch(encoder.predict(disc_x).squeeze(), disc_y,
                                                                class_weight=disc_class_weights)

                    loss = advModel.train_on_batch([_x_train, _disc_x] if not isPairData else _x_train + [_disc_x],
                                                   [_y_train, _disc_y],
                                                   class_weight=[[1] * _y_train.shape[-1], disc_class_weights])


                t2 = time()
                res = model.test_on_batch(x_val, y_val)
                val_loss = res[0]
                dis_res = discriminator.test_on_batch(encoder.predict(disc_x).squeeze(), disc_y)
                t3 = time()

                output = "Epoch %d, train[%.1f s], Dloss: %f, Dacc: %f, Mloss: %f, acc: %f, test[%.1f s]" % (
                    epoch, t2 - t1, dis_loss[0], dis_loss[1], res[0], res[1], t3 - t2)
                with open(path + "out/%s.out" % runName, "a") as myfile:
                    myfile.write(output + "\n")
                print(output)

                res = model.test_on_batch(x_test, y_test)
                output = "Test acc: %f" % res[1]
                with open(path + "out/%s.test" % runName, "a") as myfile:
                    myfile.write(output + "\n")
                print(output)
            elif modelName == "adv_bilstm2":

                for i in range(math.ceil(y_train.shape[0] / batch_size)):

                    idx = np.random.randint(0, y_train.shape[0], batch_size)
                    _x_train = x_train[idx] if not isPairData else [x_train[0][idx], x_train[1][idx]]
                    _y_train = y_train[idx]

                    idx = np.random.randint(0, len(popular_x), batch_size)
                    _popular_x = popular_x[idx]
                    idx = np.random.randint(0, len(rare_x), batch_size)
                    _rare_x = rare_x[idx]

                    _popular_x = encoder.predict(_popular_x).squeeze()
                    _rare_x = encoder.predict(_rare_x).squeeze()

                    d_loss_popular = discriminator.train_on_batch(_popular_x, popular_y)
                    d_loss_rare = discriminator.train_on_batch(_rare_x, rare_y)

                    d_loss = 0.5 * np.add(d_loss_popular, d_loss_rare)

                    idx = np.random.randint(0, len(popular_x), int(batch_size/2))
                    _popular_x = popular_x[idx]
                    idx = np.random.randint(0, len(rare_x), int(batch_size/2))
                    _rare_x = rare_x[idx]

                    _popular_rare_x = np.concatenate([_popular_x, _rare_x])

                    _popular_rare_y = np.concatenate([np.zeros(int(batch_size/2)), np.ones(int(batch_size/2))])

                    # print(advModel.predict([_x_train, _popular_rare_x]))

                    g_loss = advModel.train_on_batch([_x_train, _popular_rare_x] if not isPairData else _x_train + [_popular_rare_x],
                                                   [_y_train, _popular_rare_y])

                output = "%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %f]" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1])

                with open(path + "out/%s.res" % runName, "a") as myfile:
                    myfile.write(output + "\n")
                print(output)

                val_res = model.test_on_batch(x_val, y_val)
                test_res = model.test_on_batch(x_test, y_test)
                output = "Val acc: %f, Test acc: $f" % (val_res[1], test_res[1])
                with open(path + "out/%s.res" % runName, "a") as myfile:
                    myfile.write(output + "\n")
                print(output)


                    # if minLoss > val_loss:
            #     minLoss = val_loss
            # else:
            #     print("Early stopping")
            #     break
    else:
        # Callbacks
        # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='min')
        # cp = ModelCheckpoint(path + 'h5/%s.h5' % runName, monitor='val_loss', verbose=2, save_best_only=True,
        #                      save_weights_only=False,
        #                      mode='min', period=1)
        logger = CSVLogger(path + "out/%s.res" % runName)


        class Eval(Callback):

            def on_epoch_end(self, epoch, logs=None):
            # def on_batch_end(self, batch, logs=None):

            # def on_epoch_end(self, epoch, logs=None):
                val_res = model.test_on_batch(x_val, y_val)
                test_res = model.test_on_batch(x_test, y_test)
                output = "Val acc: %f, Test acc: $f" % (val_res[1], test_res[1])
                with open(path + "out/%s.res" % runName, "a") as myfile:
                    myfile.write(output + "\n")
                print(output)


        his = model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, shuffle=True,
                        validation_data=(x_val, y_val), callbacks=[Eval(), logger])
