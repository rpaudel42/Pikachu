# ******************************************************************************
# pikachu.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/8/21   Paudel     Initial version,
# ******************************************************************************
from CTDNE.ctdne import CTDNE
# from walk_utils import WalkUtil
from gensim.models import Word2Vec

from multiprocessing import Pool
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, Dropout
from tensorflow.keras.layers import GRU, Input, RepeatVector, TimeDistributed, Dense

import gc
import numpy as np
import pickle
from timeit import default_timer as timer

def short_term_embedding(args, node_list, idx, G):
    # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the CTDNE constructor)
    CTDNE_model = CTDNE(G, dimensions=args.dimensions, workers=4, walk_length=args.walklen,
                        num_walks=args.numwalk,
                        quiet=True)
    ctdne_model = CTDNE_model.fit(window=10, min_count=1,
                                  batch_words=4)
    # convert the wv word vectors into a numpy matrix
    node_embs = np.zeros((len(node_list), args.dimensions), dtype=np.float32)
    for i in range(len(ctdne_model.wv.vocab)):
        if ctdne_model.wv.index2word[i] in node_list:
            node_vec = ctdne_model.wv[ctdne_model.wv.index2word[i]]
            if node_vec is not None:
                node_embs[node_list.index(ctdne_model.wv.index2word[i])] = np.array(node_vec)
    return node_embs

class PIKACHU:
    def __init__(self, args, node_list, node_map, graphs):
        self.args = args
        self.mask_val = 0.
        self.node_list = node_list
        self.node_map = node_map
        self.graphs = graphs
        # self.batch_size = len(self.graphs) - self.args.lookback + 1
        # self.batch_size = 100
        self.model, self.encoder = None, None
    '''
    Initial layer is input and the masking for non-present timestep
    First couple of GRU layers create the compressed representation of the input data, the encoder.
    We then use a repeat vector layer to distribute the compressed representational
    vector across the time steps of the decoder. The final output layer of the decoder
    provides us the reconstructed input data.
    '''

    def get_model_lkbck(self):
        input = Input(batch_shape=(self.batch_size, self.args.lookback, self.args.dimensions))
        # input = Input(shape=(self.args.lookback, self.args.dimensions))
        mask = Masking(mask_value=self.mask_val)(input)
        el1 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(mask)
        do_enc = Dropout(0.5)(el1)
        # el2 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(do_enc)
        # do_enc1 = Dropout(0.5)(el2)
        encoded = GRU(self.args.dimensions, return_sequences=False, stateful=False)(do_enc)

        rp = RepeatVector(self.args.lookback)(encoded)
        dl1 = GRU(self.args.dimensions, return_sequences=True, stateful=False)(rp)
        do_de = Dropout(0.5)(dl1)
        # dl2 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(do_de)
        # do_de1 = Dropout(0.5)(dl2)
        dl3 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(do_de)
        decoded = TimeDistributed(Dense(self.args.dimensions))(dl3)

        return Model(inputs=input, outputs=decoded), Model(inputs=input, outputs=encoded)

    def autoencoder_model(self, time_step, dim):
        input = Input(shape=(time_step, dim))
        mask = Masking(mask_value=self.mask_val)(input)
        el1 = GRU(64, return_sequences=True)(mask)
        do_enc = Dropout(0.3)(el1)
        # el2 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(do_enc)
        # do_enc1 = Dropout(0.5)(el2)
        encoded = GRU(128, return_sequences=False)(do_enc)

        rp = RepeatVector(time_step)(encoded)
        dl1 = GRU(128, return_sequences=True)(rp)
        do_de = Dropout(0.3)(dl1)
        # dl2 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(do_de)
        # do_de1 = Dropout(0.5)(dl2)
        dl3 = GRU(64, return_sequences=True)(do_de)
        decoded = TimeDistributed(Dense(dim))(dl3)

        return Model(inputs=input, outputs=decoded), Model(inputs=input, outputs=encoded)

    def long_term_embedding(self, short_term_embs):
        mask_val = 0.  # padding absent nodes as 0
        total_node, time_step, dim = short_term_embs.shape
        self.model, self.encoder = self.autoencoder_model(time_step, dim)
        print(self.model.summary())
        # compile model
        self.model.compile(optimizer='adam', loss='mse')
        print("\n\nTraining Long Term Model...")
        history = self.model.fit(short_term_embs, short_term_embs, epochs=self.args.epoch,
                                 verbose=1, validation_split=0.1)

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig("fig/train_error.png")

        graph_embedding = self.model.predict(short_term_embs)
        return np.transpose(graph_embedding, (1, 0, 2))

    def learn_embedding(self, save_file):
        # '''
        print("\nGenerating Short Term embedding...")
        total_cpu = 4  # os.cpu_count()
        print("\nNumber of CPU Available: ", total_cpu)
        data_tuple = [(self.args, self.node_list, idx, G) for idx, G in enumerate(self.graphs)]
        s_time = timer()
        with Pool(total_cpu) as pool:
            short_term_embs = pool.starmap(short_term_embedding, data_tuple)
        pool.close()
        print("\nShort Term embedding Completed...   [%s Sec.]" % (timer() - s_time))
        short_term_embs = np.array(short_term_embs)
        with open('weights/short_term' + save_file, 'wb') as f:
            pickle.dump(short_term_embs, f)
        gc.collect()

        with open('weights/short_term' + save_file , 'rb') as f:
            short_term_embs = pickle.load(f)
        print("\n\nStarting Long Term Embedding...")
        # transpose to make node sequences over time e.g. (node, time, dimension)
        # previously (time, node, dimension)
        short_term_embs = np.transpose(short_term_embs, (1, 0, 2))
        s_time = timer()
        dynamic_embs = self.long_term_embedding(short_term_embs)
        print("\nLong Term Embedding Completed...   [%s Sec.]" % (timer() - s_time))
        with open('weights/long_term' + save_file , 'wb') as f:
            pickle.dump(short_term_embs, f)
        gc.collect()
