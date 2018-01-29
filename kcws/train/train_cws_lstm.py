# coding: utf-8
#!/usr/bin/python  

from __future__ import absolute_import
from __future__ import division # division
from __future__ import print_function # print（）

import numpy as np
import tensorflow as tf
import os
import time

# Define the parameters uing TF.flag that could be modified using command line
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', 'newcorpus/2014_msr_train.txt', 'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', 'newcorpus/msr_test.txt', 'Test data dir')
tf.app.flags.DEFINE_string('log_dir', "logs", 'The log  dir')
tf.app.flags.DEFINE_string('word2vec_path', 'newcorpus/vec.txt', 'the word2vec data path')
tf.app.flags.DEFINE_string('word2vec_path_2', '', 'the second word2vec data path')

tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max num of tokens per query')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'embedding size')
tf.app.flags.DEFINE_integer('embedding_size_2', 0, 'second embedding size')
tf.app.flags.DEFINE_integer('num_tags', 4, 'BMES')
tf.app.flags.DEFINE_integer('num_hidden', 100, 'hidden unit number')
tf.app.flags.DEFINE_integer('batch_size', 100, 'num example per mini batch')
tf.app.flags.DEFINE_integer('train_steps', 20000, 'trainning steps')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')


# Define the function to load the test data
def load_test_data(path):
    
    x = []
    y = []
    fp = open(path, "r")
    
    for line in fp.readlines():
        line = line.rstrip() # delete the space following the lines
        
        if not line:
            continue
        
        ss = line.split(" ")
        
        assert (len(ss) == (FLAGS.max_sentence_len) * 2) # ensure the length of test data (不足80需补齐，另一半为标注)
                
        lx = []
        ly = []
        
        for i in range(FLAGS.max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i + FLAGS.max_sentence_len]))
        
        x.append(lx)
        y.append(ly)
    
    fp.close()
    
    return np.array(x),np.array(y)        


# Define the class of LSTM_CRF
class Model:
    
    def __init__(self, embeddingSize, distinctTagNum, c2vPath, numHidden):
        
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        
        self.c2v = self.load_w2v(c2vPath, 50)
        self.words = tf.Variable(self.c2v, name = "words")
        
        if FLAGS.embedding_size_2 > 0:
            self.c2v2 = self.load_w2v(FLAGS.word2vec_path_2, FLAGS.embedding_size_2)
            self.words2 = tf.constant(self.c2v2, name = "words2")
        
        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape = [numHidden * 2, distinctTagNum],
                                     initializer = tf.truncated_normal_initializer(stddev = 0.01),
                                     name = "weights",
                                     regularizer = tf.contrib.layers.l2_regularizer(0.001)) # L2正则处理
            self.b = tf.Variable(tf.zeros([distinctTagNum], name = "bias"))
        
        self_trains_params = None
        self.inp = tf.placeholder(tf.int32, shape = [None, FLAGS.max_sentence_len], name = "input_placeholder")
        
        pass
    
    def length(self, data):
        
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices = 1)
        length = tf.cast(length, tf.int32)
        return length
    
    def inference(self, X, reuse = None, trainMode = True):
        
        word_vectors = tf.nn.embedding_lookup(self.words, X) # 按照x顺序返回self.words中的第x行，返回的结果组成tensor
        length = self.length(X) # shape为[batch_size]的vector的句子长度
        length_64 = tf.cast(length, tf.int64) # convert type
        reuse = None if trainMode else True
        
        # 如果需要第二层embedding，合并出一个新的word_vector
        if FLAGS.embedding_size_2 > 0:
            word_vectors2 = tf.nn.embedding_lookup(self.words2, X)
            word_vectors = tf.concat(2, [word_vectors, word_vectors2])
        
        # Bi-LSTM的主要参数如下：
        # 1. LSTMCell(num_cell)表示一个lstm单元暑输出的维数(100)
        # 2. 前向算法中input(word_vectors)的shape是由time_major决定，默认是false，即[batch_size, max_time, input_size].
        #    其中max_time就是80（句子最大长度），input_size就是字向量维度.
        #    回溯算法中input与前向相反,既[max_time, batch_size, input_size] - 要reverse
        # 3. sequence_length为[batch_size]的vector的句子长度
        # 4. 输出outputs为[batch_size, max_time, cell.output_size]
        
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
            
        # 连接两个三维tensor, 2表示按列连接
        # 连后size就变为[100,80,2*50]
        # reshape以后就变为[100,200]
        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.numHidden * 2])
            
        # 训练时候启用dropout, 测试时候关键时刻启用，按照50%的概率丢弃某些词（怀疑）
        if trainMode: 
            word_vectors = tf.nn.dropout(word_vectors, 0.5)
            
        # 点乘[batch_size,200]*[200,4] = [batch_size,4],100就是batch_size
        matricized_unary_scores = tf.matmul(output, self.W) + self.b
            
        # reshape之后，变为[batch_size,80,4]
        unary_scores = tf.reshape(matricized_unary_scores, [-1, FLAGS.max_sentence_len, self.distinctTagNum])
        
        return unary_scores, length
    
    def loss(self, X, Y):
        
        # CRF损失计算，训练的时候用
        # inputs是[batch_size, max_seq_len, num_tags]，即为P
        # tag_indices是[batch_size, max_seq_len]，即为Y
        # sequence_length是shape为[batch_size]的vector的句子长度
        # 输出是[batch_size]大小的vector, log-likelihood值
        # 同时输出概率转移矩阵
        P, sequence_length = self.inference(X)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        
        return loss
    
    def load_w2v(self, path, expectDim):
        
        # 处理导进来的w2v
        # 最后返回的是一个[num + 2,50]的二维矩阵，其中第一行全部为0，第二行为平均值
        
        fp = open(path, "r")
        print ("load data from: ", path)
        line = fp.readline().strip()
        ss = line.split(" ")
        
        total = int(ss[0])
        dim = int(ss[1])
        
        assert( dim == expectDim)
        
        ws = []
        mv = [0 for i in range(dim)] # 累计求和取平均值
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
        
        # 将“unk”强行调到第二行（第一行都是0）
        if second != 1:
            t = ws[1]
            ws[1] = ws[second]
            ws[second] = t
        fp.close()
        
        return np.asarray(ws, dtype = np.float32)
    
    def test_unary_score(self):
        
        P, sequence_length = self.inference(self.inp, reuse = True, trainMode = False)
        
        return P, sequence_length


# Define the function to read the train and test csv file
def read_csv(batch_size, file_name):
    
    # Output strings (e.g. filenames) to a queue for an input pipeline.
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0] for i in range(FLAGS.max_sentence_len * 2)])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size = batch_size,
                                  capacity = batch_size * 50,
                                  min_after_dequeue = batch_size)


# Define the function to evaluate the test result
def test_evaluate(sess, unary_score, test_sequence_length, transMatrix, inp, tX, tY):
    
    totalEqual = 0
    batchSize = FLAGS.batch_size
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1)/batchSize) + 1
    correct_labels = 0
    total_labels = 0
    start_time = time.time()
    
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        
        if endOff > totalLen:
            endOff = totalLen
        
        y = tY[i * batchSize:endOff]
        feed_dict = {inp: tX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run([unary_score, test_sequence_length], feed_dict)
        
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            
            # viterbi解码
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)
            
            # Evaluate word-level accuracy
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
    
    end_time = time.time()     
    accuracy = 100.0 * correct_labels / float(total_labels)
    print ("Accuracy: %.3f%%" % accuracy)
    print ("The cost of time is : " + str(end_time - start_time) + " s ")
    print ("The total_labels is " + str(total_labels))

# Define the other help function 
def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    features = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
    label = tf.transpose(tf.stack(whole[FLAGS.max_sentence_len:]))
    return features, label

def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


# Define the main function
def main(unused_argv):
    curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPath = tf.app.flags.FLAGS.train_data_path
    isTrain = True
    if not trainDataPath.startswith("/"):
        trainDataPath = curdir + "/../../" + trainDataPath
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.embedding_size, FLAGS.num_tags,
                      FLAGS.word2vec_path, FLAGS.num_hidden)
        print("train data path:", trainDataPath)
    
        # 读取训练集batch大小的feature和label，各为80大小（最大句长）的数组
        X, Y = inputs(trainDataPath)
        
        # 读取测试集的feature和label，各为80大小（最大句长）的数组
        tX, tY = load_test_data(tf.app.flags.FLAGS.test_data_path)
        
        total_loss = model.loss(X, Y)
        train_op = train(total_loss)
        
        test_unary_score, test_sequence_length = model.test_unary_score()
        
        # 创建Supervisor管理模型的分布式训练
        sv = tf.train.Supervisor(graph = graph, logdir = FLAGS.log_dir)
        with sv.managed_session(master = '') as sess:
            
            if isTrain:
                # actual training loop
                training_steps = FLAGS.train_steps
                for step in range(training_steps):
                    if sv.should_stop():
                        break
                    try:
                        _, trainsMatrix = sess.run(
                            [train_op, model.transition_params])
                        np.savetxt("crf_transition_matrix.txt", np.array(trainsMatrix))
                        # for debugging and learning purposes, see how the loss gets decremented during training steps
                        if (step + 1) % 100 == 0:
                            print("[%d] loss: [%r]" %
                                  (step + 1, sess.run(total_loss)))
                        if (step + 1) % 1000 == 0:
                            test_evaluate(sess, test_unary_score,
                                          test_sequence_length, trainsMatrix,
                                          model.inp, tX, tY)
                    except KeyboardInterrupt, e:
                        sv.saver.save(sess,
                                      FLAGS.log_dir + '/model',
                                      global_step=(step + 1))
                        raise e
                sv.saver.save(sess, FLAGS.log_dir + '/SegmentModel')

if __name__ == '__main__':
    tf.app.run()