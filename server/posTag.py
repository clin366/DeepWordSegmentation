# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-06-19 17:46:50
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-23 17:40:47
import sys
import os
import numpy as np
import tensorflow as tf
from generate_file_pos import *
from PosTagModel import *

class posTag:
    def __init__(self, char_vec_path, word_vec_path, parameters_path, crf_transition_matrix_path, tag_path, model_path):
        self.char_vec_dict = generate_dict(char_vec_path)
        self.word_vec_dict = generate_dict(word_vec_path)
        self.parameter_dict = read_parameter(parameters_path)
        self.tag_dict = generate_Tag_dict(tag_path)

        # Load the initial parameters
        distinctTagNum = self.parameter_dict["distinctTagNum"]
        num_hidden = self.parameter_dict["num_hidden"]
        embedding_word_size = self.parameter_dict["embedding_word_size"]
        embedding_char_size = self.parameter_dict["embedding_char_size"]
        char_window_size = self.parameter_dict["char_window_size"]
        self.max_sentence_len = self.parameter_dict["max_sentence_len"]
        self.max_chars_per_word = self.parameter_dict["max_chars_per_word"]
        self.batch_size = self.parameter_dict["batch_size"]

        # Load the CRF transition matrix
        self.crf_transition_matrix = np.loadtxt(crf_transition_matrix_path)

        self.graph = tf.Graph()
		# Load the BI-LSTM + CRF model
        with self.graph.as_default():
			self.model = PosTagModel(distinctTagNum, word_vec_path, char_vec_path, num_hidden, embedding_word_size, 
									 embedding_char_size, self.max_sentence_len, self.max_chars_per_word, char_window_size)
			sv = tf.train.Saver()
			self.sess = tf.Session()
			sv.restore(self.sess, model_path)
			self.text_unary_score, self.text_sequence_length = self.model.test_unary_score()

    # Define the function to get the index of the character in the c2v
    def getCharIndex(self, char):
        return self.char_vec_dict[char]

    # Define the function to get the index of the word in the w2v
    def getWordIndex(self, word):
        return self.word_vec_dict[word]

    # Define the function to help generate the char array
    def generate_char_array(self, word, index, char_list):

        word = word.decode("utf-8")

        for char in word:
            if char.encode("utf-8") in self.char_vec_dict:
                char_list[0][index] = self.getCharIndex(char.encode("utf-8"))
                index += 1
            else:
                char_list[0][index] = self.getCharIndex("<UNK>")
                index += 1

        return char_list

    # Define the function to generate the text array
    def generate_text_array(self, text):

        word_list = np.zeros([1, self.max_sentence_len], dtype = int)
        char_list = np.zeros([1, self.max_sentence_len * self.max_chars_per_word], dtype = int)
        word_index = 0
        char_index = 0

        if len(text) > self.max_sentence_len:
            text = text[:max_sentence_len]

        for word in text:
            if word in self.word_vec_dict:
                word_list[0][word_index] = self.getWordIndex(word)
                char_list = self.generate_char_array(word, char_index, char_list)
                word_index += 1
                char_index = self.max_chars_per_word * word_index
            else:
                word_list[0][word_index] = 1
                char_list = self.generate_char_array(word, char_index, char_list)
                word_index += 1
                char_index = self.max_chars_per_word * word_index

        return word_list, char_list

    # Define the function to do the posTagging work, return the tag sequence
    def posTagging_text(self, text):
        twX, tcX = self.generate_text_array(text)
		with self.graph.as_default():
			result = generate_result(self.sess, self.batch_size, self.text_unary_score, self.text_sequence_length, self.crf_transition_matrix, self.model.inp_w, self.model.inp_c, twX, tcX)
        
        for i in range(len(result)):
            result[i] = self.tag_dict[result[i]]

        return result