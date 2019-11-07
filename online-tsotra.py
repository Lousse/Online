import pandas as pd
import numpy as np
import re, os
import tensorflow as tf
import json
import warnings
import time
import shutil
import csv

from tensorflow.keras.callbacks import TensorBoard
from string import printable
from sklearn import model_selection
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from pathlib import Path
from itertools import islice
from sklearn.metrics import accuracy_score, balanced_accuracy_score


os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore")

Name="CNNLSTM-{}".format(int(time.time()))
tensorboard=TensorBoard(log_dir='logs/{}'.format(Name))

data = pd.read_csv('dataurl3.csv',encoding='latin-1', error_bad_lines=False)
data.label = [0 if i == 'good' else 1 for i in data.label]
print('Data size: ', data.shape[0])

url_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in data.url]
max_len=75
X = sequence.pad_sequences(url_tokens, maxlen=max_len)
Y = np.array(data['label'])
print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ', Y.shape)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=33)
#print(X_train.shape)
#print(len(X_train))

def print_layers_dims(model):
    l_layers = model.layers
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)

def save_model(fileModelJSON,fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)
    

def load_model(fileModelJSON,fileWeights):
    with open(fileModelJSON, 'r') as f:
         model_json = json.load(f)
         model = model_from_json(model_json)
    model.load_weights(fileWeights)
    return model

def lstm_conv(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,W_regularizer=W_reg)(main_input)  
    emb = Dropout(0.25)(emb)
        
    conv = Convolution1D(kernel_size=5, filters=256, border_mode='same')(emb)
    conv = ELU()(conv)
    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)
    
    lstm = LSTM(lstm_output_size)(conv)
    lstm = Dropout(0.5)(lstm)    
    output = Dense(1, activation='sigmoid', name='output')(lstm)
    model = Model(input=[main_input], output=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#epochs = 1
#batch_size = 32
from sklearn.metrics import classification_report

model = lstm_conv()
for i in range(len(X_train)):
    #model.fit(np.asarray([X_train[i, :]]),np.asarray([Y_train[i,]]), epochs=epochs, batch_size=batch_size, callbacks=[tensorboard])
    model.fit(np.asarray([X_train[i, :]]),np.asarray([Y_train[i,]]),  callbacks=[tensorboard])
    ''' 
    if i % 1000 == 0:
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=1) 
        print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
    '''
if i % 1000 == 0:
    predictions = model.predict(X_test)
    Y_pred=predictions.round()
        #print(predictions)
        #print(Y_test)
    print("Online Accuracy: {}".format(balanced_accuracy_score(Y_test, Y_pred)))


Chemin=''
model_name = "OCNNLSTM"
save_model(Chemin + model_name + ".json", Chemin + model_name + ".h5")
model = load_model(Chemin + model_name + ".json", Chemin + model_name + ".h5")


target_proba = model.predict(X_test, batch_size=1)
target_proba[0:10]

l_layers = model.layers
weights = l_layers[1].get_weights()
weights[0].shape


test_url_mal = "naureen.net/etisalat.ae/index2.php"
test_url_benign = "sixt.com/php/reservation?language=en_US"

url = test_url_benign
url_tokens = [[printable.index(x) + 1 for x in url if x in printable]]

max_len=75
X = sequence.pad_sequences(url_tokens, maxlen=max_len)

Y_proba = model.predict(X, batch_size=1)
def print_result(proba):
    if proba > 0.5:
        return "malicious"
    else:
        return "benign"
print("Test URL:", url, "is", print_result(Y_proba[0]))
