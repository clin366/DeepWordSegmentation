# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2017-03-19 17:46:50
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-22 16:40:47

import numpy as np
import tensorflow as tf

# Define the strucutre and variables of the BI-LSTM + CRF model
class PosTagModel:
    def __init__(self, distinctTagNum, w2vPath, c2vPath, numHidden, embedding_word_size, embedding_char_size, max_sentence_len, max_chars_per_word, char_window_size):
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.embedding_word_size = embedding_word_size
        self.embedding_char_size = embedding_char_size
        self.max_sentence_len = max_sentence_len
        self.max_chars_per_word = max_chars_per_word
        self.char_window_size = char_window_size
        self.w2v = self.load_w2v(w2vPath, self.embedding_word_size)
        self.c2v = self.load_w2v(c2vPath, self.embedding_char_size)
        self.words = tf.Variable(self.w2v, name = "words")
        self.chars = tf.Variable(self.c2v, name = "chars")


        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(
                shape = [numHidden * 2, distinctTagNum],
                initializer = tf.truncated_normal_initializer(stddev=0.01),
                name = "weights",
                regularizer = tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))

        with tf.variable_scope('CNN_Layer') as scope:
            self.filter = tf.get_variable(
                "filters_1",
                shape = [2, self.embedding_char_size, 1,
                       self.embedding_char_size],
                regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                initializer = tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

        self.trains_params = None
        self.inp_w = tf.placeholder(tf.int32,
                                    shape = [None, self.max_sentence_len],
                                    name = "input_words")
        self.inp_c = tf.placeholder(
            tf.int32,
            shape = [None, self.max_sentence_len * self.max_chars_per_word],
            name = "input_chars")

        self.loss(tf.zeros(shape = [1,self.max_sentence_len], dtype = int32), tf.zeros(shape = [1,self.max_sentence_len * self.max_chars_per_word], dtype = int32), tf.zeros(shape = [1,self.max_sentence_len], dtype = int32))

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def char_convolution(self, vecs):
        conv1 = tf.nn.conv2d(vecs,
                             self.filter, [1, 1, self.embedding_char_size, 1],
                             padding = 'VALID')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(
            conv1,
            ksize = [1, self.max_chars_per_word - self.char_window_size + 1, 1, 1],
            strides = [1, self.max_chars_per_word - self.char_window_size + 1, 1, 1],
            padding = 'SAME')
        pool1 = tf.squeeze(pool1, [1, 2])
        return pool1

    def inference(self, wX, cX, reuse=None, trainMode=True):
        word_vectors = tf.nn.embedding_lookup(self.words, wX)
        char_vectors = tf.nn.embedding_lookup(self.chars, cX)
        char_vectors = tf.reshape(char_vectors, [-1, self.max_sentence_len,
                                                 self.embedding_char_size,
                                                 self.max_chars_per_word])
        char_vectors = tf.transpose(char_vectors, perm=[1, 0, 3, 2])
        char_vectors = tf.expand_dims(char_vectors, -1)
        length = self.length(wX)
        length_64 = tf.cast(length, tf.int64)

        # do conv
        do_char_conv = lambda x: self.char_convolution(x)
        char_vectors_x = tf.map_fn(do_char_conv, char_vectors)
        char_vectors_x = tf.transpose(char_vectors_x, perm=[1, 0, 2])
        word_vectors = tf.concat([word_vectors, char_vectors_x], axis=2)
        #if trainMode:
        #  word_vectors = tf.nn.dropout(word_vectors, 0.5)
        reuse = None if trainMode else True
        with tf.variable_scope("rnn_fwbw", reuse=reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                word_vectors,
                dtype = tf.float32,
                sequence_length = length,
                scope = "RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                inputs = tf.reverse_sequence(word_vectors,
                                             length_64,
                                             seq_dim=1),
                dtype = tf.float32,
                sequence_length = length,
                scope = "RNN_backword")

        backward_output = tf.reverse_sequence(backward_output_,
                                              length_64,
                                              seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.numHidden * 2])
        if trainMode:
            output = tf.nn.dropout(output, 0.5)

        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.max_sentence_len, self.distinctTagNum])

        return unary_scores, length

    def loss(self, wX, cX, Y):
        P, sequence_length = self.inference(wX, cX)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def load_w2v(self, path, expectDim):
        fp = open(path, "r")
        line = fp.readline().strip()
        ss = line.split(" ")
        total = int(ss[0])
        dim = int(ss[1])
        assert (dim == expectDim)
        ws = []
        mv = [0 for i in range(dim)]
        second = -1
        for t in range(total):
            if ss[0] == '<UNK>':
                second = t
            line = fp.readline().strip()
            ss = line.split(" ")
            assert (len(ss) == (dim + 1))
            vals = []
            for i in range(1, dim + 1):
                fv = float(ss[i])
                mv[i - 1] += fv
                vals.append(fv)
            ws.append(vals)
        for i in range(dim):
            mv[i] = mv[i] / total
        assert (second != -1)
        # append one more token , maybe useless
        ws.append(mv)
        if second != 1:
            t = ws[1]
            ws[1] = ws[second]
            ws[second] = t
        fp.close()
        return np.asarray(ws, dtype=np.float32)

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp_w,
                                            self.inp_c,
                                            reuse=True,
                                            trainMode=False)
        return P, sequence_length