# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-06-19 17:46:50
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-23 17:40:47
import sys
import os
import re
import numpy as np
import tensorflow as tf
from generate_file_pos import *
from PosTagModel import *
from ai.botbrain.wordsegment.ttypes import PosResult

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

        # Load the BI-LSTM + CRF model
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
    def generate_char_array(self, word, index, char_list, text_index):

        for char in word:
            if char.encode("utf-8") in self.char_vec_dict:
                char_list[text_index][index] = self.getCharIndex(char.encode("utf-8"))
                index += 1
            else:
                char_list[text_index][index] = self.getCharIndex("<UNK>")
                index += 1

        return char_list

    # Define the function to generate the text array
    def generate_text_array(self, text):

        num_text = len(text);

        word_list = np.zeros([num_text, self.max_sentence_len], dtype = int)
        char_list = np.zeros([num_text, self.max_sentence_len * self.max_chars_per_word], dtype = int)

        for i in range(num_text):

            word_index = 0
            char_index = 0

            if len(text[i]) > self.max_sentence_len:
                text[i] = text[i][:self.max_sentence_len]

            for word in text[i]:
                if word in self.word_vec_dict:
                    word_list[i][word_index] = self.getWordIndex(word)
                    char_list = self.generate_char_array(word, char_index, char_list, i)
                    word_index += 1
                    char_index = self.max_chars_per_word * word_index
                else:
                    word_list[i][word_index] = 1
                    char_list = self.generate_char_array(word, char_index, char_list, i)
                    word_index += 1
                    char_index = self.max_chars_per_word * word_index

        return word_list, char_list

    # Define the function to justify the combination of the string
    def justify_combination(self, word):
        if len(word) == len(re.sub("[^a-zA-Z]", "", word)):
           return False
        else:
           return True

    # Define the function to do the posTagging work, return the tag sequence
    def posTagging_text(self, text):
        twX, tcX = self.generate_text_array(text)
        pos_result = generate_result(self.sess, self.batch_size, self.text_unary_score, self.text_sequence_length, self.crf_transition_matrix, self.model.inp_w, self.model.inp_c, twX, tcX)
        
        result = []
        for i in range(len(pos_result)):
            temp = pos_result[i][:]
            for j in range(len(temp)):
                if self.tag_dict[temp[j]] == "x" and self.justify_combination(text[i][j]):
                   temp[j] == "nz"
                else:
                   temp[j] = self.tag_dict[temp[j]]
            result.append(temp)
        
        posTag_result = []
        for i in range(len(result)):
            count = 0
            temp = []
            for word in text[i]:
                posResult = PosResult(word, result[i][count])
                count += 1
                temp.append(posResult)
            posTag_result.append(temp)
         
        return posTag_result
    
    # Define the function to do the posTagging work for the single text, return the tag sequence
    def posTagging_single_text(self, text):
        text = [text]
        posTag_result = self.posTagging_text(text)
         
        return posTag_result[0]
    
    # Define the function to do the posTagging work after segmentation for a single text
    def segment_posTagging_single_text(self, text):
        text = [text]
        posTag_result = self.segment_posTagging_text(text)
        
        return posTag_result[0]
    
    # Define the function to do the posTagging work after segmentation, return the tag sequence
    def segment_posTagging_text(self, text):
        new_text = []
        for i in range(len(text)):
            temp = []
            for j in range(len(text[i])):
                temp.append(text[i][j].decode('utf-8'))
            new_text.append(temp)

        return self.posTagging_text(new_text)