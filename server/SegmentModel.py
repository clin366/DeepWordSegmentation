# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2017-03-19 17:46:50
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-22 16:40:47

import numpy as np
import tensorflow as tf

# Define the strucutre and variables of the BI-LSTM + CRF model
class SegmentModel:
    
    def __init__(self, embeddingSize, distinctTagNum, c2vPath, numHidden, max_sentence_len):
        
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.max_sentence_len = max_sentence_len
        
        self.c2v = self.load_w2v(c2vPath, 50)
        self.words = tf.Variable(self.c2v, name = "words")

        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape = [numHidden * 2, distinctTagNum],
                                     initializer = tf.truncated_normal_initializer(stddev = 0.01),
                                     name = "weights",
                                     regularizer = tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name = "bias"))
        
        self_trains_params = None
        self.inp = tf.placeholder(tf.int32, shape = [None, self.max_sentence_len], name = "input_placeholder")
        self.loss(tf.zeros(shape=[1,80], dtype = tf.int32), tf.zeros(shape=[1,80], dtype = tf.int32))
        
    def length(self, data):
        
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length
    
    def inference(self, X, reuse = None):
        
        word_vectors = tf.nn.embedding_lookup(self.words, X) 
        length = self.length(X) 
        length_64 = tf.cast(length, tf.int64)
        
        with tf.variable_scope("rnn_fwbw", reuse = reuse) as scope:
            
            forward_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(self.numHidden),
                                                  word_vectors, 
                                                  dtype = tf.float32,
                                                  sequence_length = length,
                                                  scope = "RNN_forward")
            
            backward_output_, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(self.numHidden),
                                                   inputs = tf.reverse_sequence(word_vectors, length_64, seq_dim = 1),
                                                   dtype = tf.float32,
                                                   sequence_length = length,
                                                   scope = "RNN_backward")
            
        backward_output = tf.reverse_sequence(backward_output_,
                                                  length_64,
                                                  seq_dim = 1)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.numHidden * 2])
        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        unary_scores = tf.reshape(matricized_unary_scores, [-1, self.max_sentence_len, self.distinctTagNum])
        
        return unary_scores, length
    
    def loss(self, X, Y):
        
        P, sequence_length = self.inference(X)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        
        return loss
    
    def load_w2v(self, path, expectDim):
        
        fp = open(path, "r")
        line = fp.readline().strip()
        ss = line.split(" ")
        
        total = int(ss[0])
        dim = int(ss[1])
        
        assert( dim == expectDim)
        
        ws = []
        mv = [0 for i in range(dim)]
        second = -1
        
        for t in range(total):
            if ss[0] == '<UNK>':
                second = t
            
            line = fp.readline().strip()
            ss = line.split(" ")
            
            assert(len(ss) == (dim + 1))
            
            vals = []
            
            for i in range(1, dim + 1):
                fv = float(ss[i])
                mv[i - 1] += fv
                vals.append(fv)
            ws.append(vals)
            
        for i in range(dim):
            mv[i] = mv[i]/total
        
        assert(second != -1)
        
        ws.append(mv)
        
        if second != 1:
            t = ws[1]
            ws[1] = ws[second]
            ws[second] = t
        fp.close()
        
        return np.asarray(ws, dtype = np.float32)
    
    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp, reuse = True)
        return P, sequence_length