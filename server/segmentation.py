# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-06-19 17:46:50
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-23 17:40:47
import sys
import os
import numpy as np
import tensorflow as tf
from generate_file_segment import *
from SegmentModel import *

class segmentation:
    def __init__(self, vec_path, parameters_path, crf_transition_matrix_path, model_path):
        self.vec_dict = generate_vec_dict(vec_path)
        self.parameter_dict = read_parameter(parameters_path)

        # Load the initial parameters
        self.max_sentence_len = self.parameter_dict["max_sentence_len"]
        embedding_size = self.parameter_dict["embedding_size"]
        num_tags = self.parameter_dict["num_tags"]
        num_hidden = self.parameter_dict["num_hidden"]

        # Load the CRF transition matrix
        self.crf_transition_matrix = np.loadtxt(crf_transition_matrix_path)

        self.graph = tf.Graph()
        # Load the BI-LSTM + CRF model
        with self.graph.as_default():
            self.model = SegmentModel(embedding_size, num_tags, vec_path, num_hidden, self.max_sentence_len)
            sv = tf.train.Saver()
            self.sess = tf.Session();
            sv.restore(self.sess, model_path)
            self.text_unary_score, self.text_sequence_length = self.model.test_unary_score()

    # Define the function to get the indxe of the character in the w2v
    def getIndex(self, char):
        return self.vec_dict[char]

    # Define the function to generate the text array
    def generate_text_array(self, text):

        num_text = len(text)
        char_list = np.zeros([num_text, self.max_sentence_len], dtype = int)

        for i in range(len(text)):
            count = 0
            for char in text[i]:
                if char in self.vec_dict:
                    char_list[i][count] = self.getIndex(char)
                    count += 1
                else:
                    char_list[i][count] = self.getIndex("<UNK>")
                    count += 1

        return char_list

    # Define the function to do the segmentation work, return tag sequence with symbol
    def segment_text_without_filter(self, text):
        tX = self.generate_text_array(text)

        with self.graph.as_default():
            result = generate_result(self.sess, self.text_unary_score, self.text_sequence_length, self.crf_transition_matrix, self.model.inp, tX)
                    
        return result

    # Define the function to return the char result
    def generate_char_result(self, text, result):
        char_result = []
        assert len(text) == len(result)

        char = ""
        for i in range(len(result)):
            if result[i] == 0:
                char_result.append(text[i])
            elif result[i] == 3:
                char += text[i]
                char_result.append(char)
                char = ""
            else:
                char += text[i]

        if char != "":
           char_result.append(char)

        return char_result

    def generate_final_result_single_text(self, text):
        text = [text]
        final_result = self.generate_final_result(text)

        return final_result[0]

    def generate_final_result(self, text):

        new_text = []
        for i in range(len(text)):
            new_text.append((text[i][:80]).replace(" ",""))

        result = self.segment_text_without_filter(new_text)
        num_text = len(result)
        final_result = []

        for i in range(num_text):
            final_result.append(self.generate_char_result(new_text[i], result[i]))

        return final_result