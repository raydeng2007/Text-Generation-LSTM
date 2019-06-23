from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,load_model
import keras.utils as ku
import pandas as pd
import json
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import flask
from flask import Flask, render_template,url_for,request,copy_current_request_context
import time
from flask_socketio import SocketIO, emit


#creating instance of the class

app=Flask(__name__)
socketio = SocketIO(app)
global m
m = load_model('model')
m._make_predict_function()
global n_vocab
global int_to_char
int_to_char = {0: ' ',
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
n_vocab = 37

is_text_box_empty=True
#to tell flask what url should trigger the function index()
@app.route('/',methods=['GET','POST'])
@app.route('/index',methods=['GET','POST'])
def index():
    return flask.render_template('index.html',async_mode=socketio.async_mode)


@socketio.on('input')
def test_message(input_words):
    global is_text_box_empty
    input_words = input_words['data']
    if is_text_box_empty:

        is_text_box_empty = False
    else:
        socketio.emit('clear')
        #global is_text_box_empty
        is_text_box_empty = True

    if len(input_words) < 100:
        # this needs to be at least 100
        print('yo there something wrong here lil bro')
    else:

        def sample(preds, temperature=0.8):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds[0]).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        pattern = []
        input_words = input_words[-100:]
        for char in input_words:
            try:
                pattern.append(self.char_to_int[char])
            except:
                pattern.append(0)
        output = ''
        for i in range(800):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = m.predict(x, verbose=0)
            index = sample(prediction)
            result = int_to_char[index]
            output = output + result
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            emit('my_response', {'data': result})

if __name__ == '__main__':
    socketio.run(app)







