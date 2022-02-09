#!/usr/bin/python3
# coding: utf-8

'''
Author:  Daniel Waddington <daniel.waddington@ibm.com> 2021
Changed: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''
# this is a very simple word vectorization implementation
import numpy as np
import urllib.request, urllib.parse, urllib.error
import bs4 as bs
import time
import pickle


filename = "data/2_files.pickled"
file_data  = pickle.load(open(filename, "rb"))
print ("The number of files are: " + str(len(file_data)))


def get_all_words(file_content):
    import nltk
    import re
    import email
    from nltk.tokenize import word_tokenize, sent_tokenize
    msg = email.message_from_bytes(file_content)
    article_text = msg.get_payload().replace("\n", " ")
    all_sentences = nltk.sent_tokenize(article_text)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    return all_words


def add_words_to_symtab(new_words, word_to_id, id_to_word):
    for new_word in new_words:
        if new_word in word_to_id:
            continue
        i = len(word_to_id)
        word_to_id[new_word] = i
        id_to_word[i] = new_word


def do_work(count, file_data):
    # build bi-directional symbol table mapping words to a value counting from 0
    word_to_id={}
    id_to_word={}

    tic = time.perf_counter()
    for i in range(count):
        all_words = get_all_words(file_data[i])
        for sentence in all_words:
            add_words_to_symtab(sentence,word_to_id,id_to_word)
            
    vector_length = len(word_to_id)
    row_count = count
    #print(vector_length, row_count)
    toc = time.perf_counter()    
    print(f"Symbol table built in {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    word_vectors = np.zeros((row_count,vector_length,), dtype=np.uint8)
    # build vectors
    for i in range(0,count):
        all_words = get_all_words(file_data[i])
        for sentence in all_words:
            for word in sentence:
                word_vectors[i][word_to_id[word]] += 1
    toc = time.perf_counter()
    print(f"Vectors result shape {word_vectors.shape} complete in {toc - tic:0.4f} seconds")


count = len(file_data)
print ("perform_segmentation on " + str(count) + " images")
do_work(count, file_data)
