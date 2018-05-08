# data preprocessing
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

def clean_str(string):
    
    string = re.sub(r"/s", " ", string)
    string = re.sub(r"[^A-Za-z0-9()'`]", " ", string)
    #,!?\'\`
    string = re.sub(r" . ", " ", string)
    string = re.sub(r"[0-9]", " ", string)
    string = re.sub(r" [!?\'\`]", "", string)
    
    
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def load_data_and_labels():
    
    data_file_path = 'all_data_column_titles.csv'
    #data_file_path = 'two_K_data.csv'
    sarcasm_data = pd.read_csv(data_file_path, encoding = "ISO-8859-1")
    sarcasm_data['text'] = sarcasm_data['text'].apply(clean_str)
    text = sarcasm_data.text
    labels = sarcasm_data.label   
    train_X, val_X, train_y, val_y = train_test_split(text, labels,random_state = 0)
    return [train_X, val_X, train_y, val_y]


def give_labels_and_pad_predictors(train_X, val_X, train_y, val_y):
    # calculate the max len of a sentence/ input dim
    max_length = 0
    for line in train_X:
        if len(line.split()) > max_length:
            max_length = len(line.split())
        if len(line.split()) > 1644:
            print(line)
            print('------------------------')
    for line in val_X:
        if len(line.split()) > max_length:
            max_length = len(line.split())
        if len(line.split()) > 1644:
            print(line)
            
    # prepare tokenizer
    # give ids to words in the list in fitting the the tokenizer
    t = Tokenizer()
    t.fit_on_texts(train_X.tolist())
    # t.fit_on_texts(train_X)
    # t.word_index.items() gives the words that are encoded in the tokenizer
    vocab_size = len(t.word_index) + 1
    
    # representing a sentence by the ids of the words forming it
    # encode the documents
    encoded_train_X = t.texts_to_sequences(train_X)
    encoded_val_X = t.texts_to_sequences(val_X)    
    # padding sentence sequences with 0 to reach the max length             
    padded_train_X = pad_sequences(encoded_train_X, maxlen=max_length, padding='post')
    padded_val_X = pad_sequences(encoded_val_X, maxlen=max_length, padding='post')
        
    return [t, max_length, vocab_size, encoded_train_X, padded_train_X, padded_val_X] 
    
def load_GloVe():    
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('glove.6B.50d.txt')
    for line in f:
       # the -0.038194 -0.24487 0.72812 -0.39961 ................
    	values = line.split()
    	word = values[0]
    	coefs = asarray(values[1:])
    	embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index

def creating_embedding_matrix(t, vocab_size, embeddings_index, encoded_train_X):
    #flatten 2D array, produce 1D array representing ALL sentences/rows in the dataset
    # to be able to iterate over it in NEXT loop
    flat_encoded_train_X = []
    for item in encoded_train_X:
        for x in item:
            flat_encoded_train_X.append(x)
    
    # number of loop iteratoins
    # create a weight matrix for words in training docs**
    embedding_matrix = zeros((vocab_size, 50))
    for word, i in t.word_index.items():
        if i in flat_encoded_train_X:            
            #print('word %s  i %f' % (word, i))
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix
       
