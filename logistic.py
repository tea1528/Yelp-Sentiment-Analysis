import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from nltk.stem import PorterStemmer
import re
import spacy
import pickle


stop_words = set(stopwords.words('english') + list(string.punctuation))

def tokenize(text):
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


NLP = spacy.load('en')

def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\.+", " ", comment)
    comment = re.sub(r"\?+", "?", comment)
    comment = comment.replace("n't", "not")
    comment = comment.lower()
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]

def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a word (sparse) matrix, type: scipy.sparse.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    '''
    # use of linked list to create spare matrix
    data_matrix = sparse.lil_matrix((len(data), len(vocab_dict)))

    # YOUR CODE HERE
    for i, doc in enumerate(data):
        for word in doc:
            # dict.get(key, -1)
            # if the word in the vocab_dic, return the value
            # else return -1
            word_idx = vocab_dict.get(word, -1)
            if word_idx != -1:
                data_matrix[i, word_idx] += 1

    # csr: row based format, better for matrix multiplication
    # to speed up when computing
    data_matrix = data_matrix.tocsr()

    return data_matrix


def read_data(file_name, vocab=None, tfidf=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenizer)

    stemmer = PorterStemmer()
    df['words'] = df['words'].apply(lambda x: [stemmer.stem(y) for y in x])

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)

    # dictionary of vocab : index_num
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    count = get_bagofwords(df['words'], vocab_dict)

    if tfidf is None:
        tfidf_vect = TfidfTransformer()
    else:
        tfidf_vect = tfidf

    data_matrix = tfidf_vect.fit_transform(count)

    return df['id'], df['label'], data_matrix, tfidf_vect, vocab

def evaluate(y_true, y_pre):
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1

if __name__ == '__main__':
    train_id_list, train_data_label, train_data_matrix, tfidf, vocab = read_data("all/train.csv")
    print("Training Set Size:", len(train_id_list))
    test_id_list, _, test_data_matrix, _, _ = read_data("all/test.csv", vocab=vocab, tfidf=tfidf)
    print("Test Set Size:", len(test_id_list))

    # and later you can load it
    # with open('filename.pkl', 'rb') as f:
    #     clf = pickle.load(f)

    # grt_label = pd.read_csv("all/sample_submission.csv")
    # acc, precision, recall, f1 = evaluate(grt_label["pred"], test_data_pre)
    # print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    # clf = sn.MultinomialNB()
    clf = LogisticRegressionCV(cv=10, multi_class='multinomial', solver='lbfgs',
                               n_jobs=-1, random_state=0)
    clf.fit(train_data_matrix, train_data_label)
    predicted = clf.predict(test_data_matrix)

    # now you can save it to a file
    with open('log_tfidf.pkl', 'wb') as f:
        pickle.dump(clf, f)



    print("Saving predicted result on test set into a csv file...")
    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list
    sub_df["pred"] = predicted
    sub_df.to_csv("log.csv", index=False)
