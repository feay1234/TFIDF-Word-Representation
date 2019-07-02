import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback

import argparse
from time import time
from Dataset import *
from FRAGE import FRAGE
from Models import *
from datetime import datetime
from sklearn.utils import class_weight
import math
from utils import get_pretrain_embeddings, save2file


def parse_args():
    parser = argparse.ArgumentParser(description="Run TFIDF-Word-Representation")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="adv_bilstm")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="SICK_E")

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

    parser.add_argument('--dm', type=str, default="idf",
                        help='Discriminator mode: tf or idf')

    parser.add_argument('--mode', type=int, default="1",
                        help='Mode:')

    parser.add_argument('--bs', type=int, default=32,
                        help='Batch Size:')
    parser.add_argument('--w', type=float, default=0.1,
                        help='Weight:')
    parser.add_argument('--pp', type=float, default=0.2,
                        help='Popularity Percentage:')

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
    pop_percent = args.pp


    isPairData = True if dataset in ["QQP", "MRPC", "SICK_R", "SICK_E", "SNLI", "STS"] else False

    x_train, y_train, x_val, y_val, x_test, y_test, word_index, class_num, maxlen = get_datasets(path, dataset,
                                                                                                 max_words,
                                                                                                 maxlen, isPairData)
    embedding_layer = get_pretrain_embeddings(path, max_words, emb_dim, maxlen, word_index)


    if modelName == "bilstm":
        model = get_bilstm_maxpool(dim, max_words, maxlen, embedding_layer, class_num, isPairData)
        runName = "%s_%s_d%d_w%d_ml%d_%s" % (dataset,
                                             modelName, dim, max_words, maxlen,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    elif modelName == "adv_bilstm":
        run = FRAGE(dim, emb_dim, max_words, maxlen, embedding_layer, class_num, isPairData,weight)
        runName = "%s_%s_%s_d%d_w%d_ml%d_w%.3f_pp%.3f_%s" % (dataset,
                                                          modelName, discMode, dim, max_words, maxlen, weight,
                                                          pop_percent,
                                                          datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        run.init(x_train, discMode, isPairData, batch_size, pop_percent)

    save2file(path + "out/%s.res" % runName, runName)

    if "adv" in modelName:
        minMSE = 9999999
        maxACC = -999999
        for epoch in range(epochs):

            t1 = time()
            output = run.train(x_train, y_train, batch_size, epoch, isPairData)
            t2 = time()
            save2file(path + "out/%s.out" % runName, output)

            # Eval
            val_res = model.test_on_batch(x_val, y_val)
            test_res = model.test_on_batch(x_test, y_test)
            output = "Val acc: %f, Test acc: %f" % (val_res[1], test_res[1])
            save2file(path + "out/%s.res" % runName, output)

            # Early stopping strategy: MSE and ACC
            if class_num == 1:
                if minMSE > val_res[1]:
                    minMSE = val_res[1]
                else:
                    print("Early stopping")
                    break
            else:
                if maxACC < val_res[1]:
                    maxACC = val_res[1]
                else:
                    print("Early stopping")
                    break
    else:
        # Callbacks
        # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='min')
        # cp = ModelCheckpoint(path + 'h5/%s.h5' % runName, monitor='val_loss', verbose=2, save_best_only=True,
        #                      save_weights_only=False,
        #                      mode='min', period=1)
        logger = CSVLogger(path + "out/%s.out" % runName)


        class Eval(Callback):

            def on_epoch_end(self, epoch, logs=None):
                # def on_batch_end(self, batch, logs=None):

                # def on_epoch_end(self, epoch, logs=None):
                val_res = model.test_on_batch(x_val, y_val)
                test_res = model.test_on_batch(x_test, y_test)
                output = "Val acc: %f, Test acc: %f" % (val_res[1], test_res[1])
                save2file(path + "out/%s.res" % runName, output)


        his = model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, shuffle=True,
                        validation_data=(x_val, y_val), callbacks=[Eval(), logger])
