# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

import re
import nltk
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english') + list(string.punctuation))
"""
def tokenizer(comment):
    # preprocessing using regular expression
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    return [
        x.text for x in nltk.word_tokenize(comment) if x.text != " "]
"""
def tokenizer(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    # YOUR CODE HERE
    for word in nltk.word_tokenize(text):
        word = word.lower()
        # remove stop_words (commonly used meaningless words) and numbers
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)

    return tokens

def load_dataset(embed_len=300, batch_size=32):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
#     tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()
    
    fields = [('stars', LABEL), ('text', TEXT)]
    
    train_data = data.TabularDataset(
        path='data/mod_train.csv', format='csv', 
        skip_header=True,
        fields=fields)
    

    valid_data = data.TabularDataset(
        path='data/mod_valid.csv', format='csv', 
        skip_header=True,
        fields=fields)

    test_data = data.TabularDataset(
        path='data/test.csv', format='csv', 
        skip_header=True,
        fields=fields)
    
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=embed_len))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    print ("Label Mapping: ", LABEL.vocab.stoi)
    print ("Most frequent: ", TEXT.vocab.freqs.most_common(20))

    # train_data, valid_data = train_data.split(split_ratio=0.8) # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
