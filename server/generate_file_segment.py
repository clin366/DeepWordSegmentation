# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-6-20 14:17:53
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-22 15:20:11
import time
import string
import numpy as np
import tensorflow as tf

# Define the function to generate the char vector dictionary
def generate_vec_dict(path):

  vec_result = open(path, "r")
  vec_dict = {}
  index = 0

  while True:
    line = vec_result.readline()

    if not line:
      break

    line = line.split(" ")
    vec_dict[line[0]] = index - 1
    index += 1

  vec_result.close()
  return vec_dict

# Define the function to read the parameters
def read_parameter(path):

    parameter_file = open(path, "r")
    parameter = {}

    while True:
        line = parameter_file.readline()

        if not line:
            break

        line = line.split(",")
        parameter[line[0]] = int(line[1])

    parameter_file.close()
    return parameter

# Define the function to filter the symbol in char result
def filter_symbol(char_list):

    new_list = []
    symbol_chinese = "。。×-★！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏." 

    for word in char_list:
        if word not in symbol_chinese and word not in string.punctuation:
            new_list.append(word)

    return new_list

# Define the function to filter the symbol    
def generate_result_without_symbol(sequence, text):

    text = text.decode("UTF-8")
    char = []

    for word in text:
        if word != "<" and word != ">" and word != " " and word != "\n" and word != "\r":
                char.append(word.encode('UTF-8'))

    symbol_chinese = "。。×-★！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."          
    new_sequence = []

    for i in range(len(char)):
        assert len(char) == len(sequence)

        if char[i] in symbol_chinese or char[i] in string.punctuation:
            sequence[i] = "P"

    for i in range(len(char)):
        if sequence[i] == "P":
            continue
        else:
            new_sequence.append(int(sequence[i]))

    return new_sequence

# Define the function to do the segmentation work
def generate_result(sess, unary_score, test_sequence_length, transMatrix, inp, tX, tY):
    
    totalEqual = 0
    batchSize = 100
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1)/batchSize) + 1
    
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
            
            # viterbi decode
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)

    return viterbi_sequence