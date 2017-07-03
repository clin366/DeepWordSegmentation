# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-6-20 14:17:53
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-22 15:20:11
import time
import numpy as np
import tensorflow as tf

def generate_Tag_dict(path):

  tag_result = open(path, "r")
  tag_dict = {}

  while True:
    line = tag_result.readline()

    if not line:
      break

    temp = line.split(",")

    try :
      tag_dict[int(temp[1][:2])] = temp[0]
    except:
      tag_dict[int(temp[1][:1])] = temp[0]

  return tag_dict
  
# Define the function to generate the vector dictionary
def generate_dict(path):

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

def generate_result(sess, batchSize, unary_score, test_sequence_length, transMatrix, inp_w, inp_c, twX, tcX):
    
    totalLen = twX.shape[0]
    numBatch = int((twX.shape[0] - 1) / batchSize) + 1
    result = []

    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        feed_dict = {inp_w: twX[i * batchSize:endOff], inp_c: tcX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run([unary_score, test_sequence_length], feed_dict)

        for tf_unary_scores_, sequence_length_ in zip(unary_score_val, test_sequence_length_val):
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)
            result.append(viterbi_sequence)

    return result