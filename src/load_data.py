# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import yaml
import re
import string
import spacy

NLP = spacy.load('en_core_web_md')
config_filepath = 'config.yaml'
with open(config_filepath) as f:
    config = yaml.load(f)

def tokenizer(comment):
    # preprocessing using regular expression
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!\.\,;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    return [
        x.text for x in NLP.tokenizer(comment) if x.text != ' ']


def load_dataset(embed_len=300, batch_size=32):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fix_length which
                 will pad each sequence to have a fix length of 400.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    """
    
    #tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, fix_length=400)
    LABEL = data.LabelField()
    
    fields = [('stars', LABEL), ('text', TEXT)]
    
    train_data = data.TabularDataset(
        path=config['training']['train_path'], format='csv', 
        skip_header=True,
        fields=fields)
    
    valid_data = data.TabularDataset(
        path=config['training']['train_path'], format='csv', 
        skip_header=True,
        fields=fields)

    TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=embed_len))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    print ("Label Mapping: ", LABEL.vocab.stoi)
    print ("Most Frequent: ", TEXT.vocab.freqs.most_common(20))

    # train_data, valid_data = train_data.split(split_ratio=0.8) # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)
    mapping = LABEL.vocab.stoi

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, mapping
