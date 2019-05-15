from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,load_model
import keras.utils as ku
import re
import json
import pandas as pd
import csv
import json
from keras.preprocessing.text import text_to_word_sequence
import sys
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template,url_for,request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    m = model()
    print('loaded model')
    input_words = request.form['comment']
    if len(input_words) < 100:
        # this needs to be at least 100
        print('yo there something wrong here lil bro')
    else:
        pattern = []
        input = input_words
        input_words = input_words[-100:]
        for char in input_words:
            try:
                pattern.append(self.char_to_int[char])
            except:
                pattern.append(0)
        output = ''
        for i in range(1000):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(m.n_vocab)
            prediction = m.model.predict(x, verbose=0)
            index = m.sample(prediction)
            # index =  np.argmax(prediction)
            result = m.int_to_char[index]
            seq_in = [m.int_to_char[value] for value in pattern]
            output = output+result
            pattern.append(index)
            pattern = pattern[1:len(pattern)]


    return render_template('result.html', prediction=output, input=input)


class model():
    def __init__(self):
        self.model = load_model('model')
        self.char_to_int = {' ': 0,
                             '0': 1,
                             '1': 2,
                             '2': 3,
                             '3': 4,
                             '4': 5,
                             '5': 6,
                             '6': 7,
                             '7': 8,
                             '8': 9,
                             '9': 10,
                             'a': 11,
                             'b': 12,
                             'c': 13,
                             'd': 14,
                             'e': 15,
                             'f': 16,
                             'g': 17,
                             'h': 18,
                             'i': 19,
                             'j': 20,
                             'k': 21,
                             'l': 22,
                             'm': 23,
                             'n': 24,
                             'o': 25,
                             'p': 26,
                             'q': 27,
                             'r': 28,
                             's': 29,
                             't': 30,
                             'u': 31,
                             'v': 32,
                             'w': 33,
                             'x': 34,
                             'y': 35,
                             'z': 36}

        self.int_to_char = {0: ' ',
                             1: '0',
                             2: '1',
                             3: '2',
                             4: '3',
                             5: '4',
                             6: '5',
                             7: '6',
                             8: '7',
                             9: '8',
                             10: '9',
                             11: 'a',
                             12: 'b',
                             13: 'c',
                             14: 'd',
                             15: 'e',
                             16: 'f',
                             17: 'g',
                             18: 'h',
                             19: 'i',
                             20: 'j',
                             21: 'k',
                             22: 'l',
                             23: 'm',
                             24: 'n',
                             25: 'o',
                             26: 'p',
                             27: 'q',
                             28: 'r',
                             29: 's',
                             30: 't',
                             31: 'u',
                             32: 'v',
                             33: 'w',
                             34: 'x',
                             35: 'y',
                             36: 'z'}

        self.n_vocab = 37

    def sample(self,preds, temperature=0.8):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds[0]).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)











def result():
    m = model()
    if request.method == 'POST':
        word_to_predict = request.form.to_string()
        result = m.predict(word_to_predict)


if __name__ == '__main__':

    app.run(debug=True)







