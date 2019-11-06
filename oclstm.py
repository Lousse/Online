from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from imblearn.datasets import make_imbalance
import numpy as np
from clstm import clstm,clstm_THS,X,Y
import sklearn
from sklearn.model_selection import train_test_split

vocab_size=100
#input_dim = 75
embedding_dim = 100
hidden_dim = 256
output_dim = 32
n_layers = 2
bidirectional = True
DROPOUT = 0.5
class_num=2
batch_size=32
dropout=0.5

model = clstm(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,class_num,batch_size,output_dim, dropout)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

for i in range(len(X_train)):
  model.partial_fit(np.asarray([X_train[i, :]]), np.asarray([y_train[i]]))
  
  if i % 1000 == 0:
    predictions = model.predict(X_test)
    print("Online Accuracy: {}".format(balanced_accuracy_score(y_test, predictions)))
