import collections
import json
import random
import string
import math
import pandas as pd

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from keras.preprocessing import sequence
from torch.nn.parameter import Parameter
from mab import algs
from string import printable
from torchtext import data
from torch import cuda


SEED = 1234

torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

'''
printable = string.printable
text = open('trainurl.csv', 'r').read()
pruned_text = ''
for line in text:
    if text in printable:
        pruned_text+=text  
        
X = data.Field(pruned_text)
Y = data.LabelField()
print(X,Y)
'''

data = pd.read_csv('data.csv',encoding='latin-1', error_bad_lines=False)
data.label = [0 if i == 'good' else 1 for i in data.label]
print('Data size: ', data.shape[0])

url_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in data.url]
max_len=75
X = sequence.pad_sequences(url_tokens, maxlen=max_len)
Y = np.array(data['label'])
print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ', Y.shape)

class clstm(nn.Module):
    def __init__(self, vocab_size,max_num_hidden_layers,embedding_dim, n_classes,n_filters,filter_size,
                  output_dim,dropout, batch_size=1,b=0.99, n=0.01, s=0.2, use_cuda=False):
        super(clstm, self).__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.vocab_size = vocab_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.n_classes = n_classes
        self.batch_size = batch_size       
        
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)

        self.embedding = []
        self.embedding = nn.Embedding(vocab_size, embedding_dim,dropout)
        self.sigmoid=nn.Sigmoid()
        
        self.conv=[]
        self.conv = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size,embedding_dim))
        self.dropout = nn.Dropout()
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool1d(kernel_size=(filter_size,embedding_dim)) 
        
        self.lstm=[]
        self.lstm = nn.LSTM(n_filters, max_num_hidden_layers, dropout)
        
        self.outputs=[]
        self.fc = nn.Linear(max_num_hidden_layers, output_dim)
        self.dropout = nn.Dropout()

        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers).fill_(1 / (self.max_num_hidden_layers + 1)),requires_grad=False).to(self.device)
        self.loss_array = []
      

    def zero_grad(self):
        for i in range(self.max_num_hidden_layers):
            self.outputs[i].weight.grad.data.fill_(0)
            self.embedding[i].weight.grad.data.fill_(0)
            self.conv[i].bias.grad.data.fill_(0)
            self.lstm[i].bias.grad.data.fill_(0)

    def update_weights(self, X, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)
        predictions_per_layer = self.forward(X)
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
            losses_per_layer.append(loss)

        w = []
        b = []

        for i in range(len(losses_per_layer)):
            losses_per_layer[i].backward(retain_graph=True)
            self.outputs[i].weight.data -= self.n * self.alpha[i] * self.outputs[i].weight.grad.data
            self.outputs[i].bias.data -= self.n * self.alpha[i] * self.outputs[i].bias.grad.data
            w.append(self.alpha[i] * self.embedding[i].weight.grad.data)
            w.append(self.alpha[i] * self.conv[i].weight.grad.data)
            w.append(self.alpha[i] * self.lstm[i].weight.grad.data)
            b.append(self.alpha[i] * self.embedding[i].bias.grad.data)
            b.append(self.alpha[i] * self.conv[i].bias.grad.data)
            b.append(self.alpha[i] * self.lstm[i].bias.grad.data)
            self.zero_grad()

        for i in range(1, len(losses_per_layer)):
            self.embedding[i].weight.data -= self.n * torch.sum(torch.cat(w[i:]))
            self.conv[i].weight.data -= self.n * torch.sum(torch.cat(w[i:]))
            self.lstm[i].weight.data -= self.n * torch.sum(torch.cat(w[i:]))
            self.embedding[i].bias.data -= self.n * torch.sum(torch.cat(b[i:]))
            self.conv[i].bias.data -= self.n * torch.sum(torch.cat(b[i:]))
            self.lstm[i].bias.data -= self.n * torch.sum(torch.cat(b[i:]))

        for i in range(len(losses_per_layer)):
            self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / self.max_num_hidden_layers)

        z_t = torch.sum(self.alpha)

        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        if show_loss:
            real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers, self.batch_size, 1), predictions_per_layer), 0)
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
            self.loss_array.append(loss)
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = torch.Tensor(self.loss_array).mean().cpu().numpy()
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
                self.loss_array.clear()
                
                
    def forward(self,X):
        '''
        module_input=torch.Tensor(X).long().to(self.device)
        #module_input = torch.from_numpy(X).float().to(self.device)
        x = self.embedding(X)
        x = self.sigmoid(X)
        x = self.conv(X)
        x = self.dropout(X)
        x = self.relu(X)
        x = self.maxpool(X)
        x = self.lstm(X)
        x = self.fc(X)
        x = self.dropout(X)
        x = self.alpha(X)
        return module_input * X   
        '''
        
        hidden_connections = []
        X = torch.Tensor(X).long().to(self.device)
        x1 = F.sigmoid(self.embedding(X))
        hidden_connections.append(x1)
        x2 = F.sigmoid(self.conv(X))
        hidden_connections.append(x2)
        x3 = F.sigmoid(self.lstm(X))
        hidden_connections.append(x3)

        output_class = []
        for i in range(self.max_num_hidden_layers):
            output_class.append(self.outputs[i](hidden_connections[i]))
        pred_per_layer = torch.stack(output_class)
        return pred_per_layer     

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception("Wrong dimension for this X data. It should have only two dimensions.")

    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception("Wrong dimension for this Y data. It should have only one dimensions.")

    def partial_fit_(self, X_data, Y_data, show_loss=True):
        self.validate_input_X(X_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, Y_data, show_loss)

    def partial_fit(self, X_data, Y_data, show_loss=True):
        self.partial_fit_(X_data, Y_data, show_loss)

    def predict_(self, X_data):
        self.validate_input_X(X_data)
        return torch.argmax(torch.sum(torch.mul(self.alpha.view(self.max_num_hidden_layers, 1).repeat(1, len(X_data)).view(self.max_num_hidden_layers, len(X_data), 1), self.forward(X_data)), 0), dim=1).cpu().numpy()

    def predict(self, X_data):
        pred = self.predict_(X_data)
        return pred

    def export_params_to_json(self):
        state_dict = self.state_dict()
        params_gp = {}
        for key, tensor in state_dict.items():
            params_gp[key] = tensor.cpu().numpy().tolist()
        return json.dumps(params_gp)

    def load_params_from_json(self, json_data):
        params = json.loads(json_data)
        o_dict = collections.OrderedDict()
        for key, tensor in params.items():
            o_dict[key] = torch.tensor(tensor).to(self.device)
        self.load_state_dict(o_dict)


class clstm_THS(clstm):
    def __init__(self, vocab_size,max_num_hidden_layers,embedding_dim, n_classes,n_filters,filter_size,
                  dropout, batch_size,b=0.99, n=0.01, s=0.2, use_cuda=False):
        super().__init__(vocab_size,max_num_hidden_layers,embedding_dim, n_classes,n_filters,filter_size,
                  dropout, batch_size, b=b, n=n, s=s,use_cuda=use_cuda)
        self.e = Parameter(torch.tensor(e), requires_grad=False)
        self.arms_values = Parameter(torch.arange(n_classes), requires_grad=False)
        self.explorations_mab = []

        for i in range(n_classes):
            self.explorations_mab.append(algs.ThompsomSampling(len(e)))

    def partial_fit(self, X_data, Y_data, exp_factor, show_loss=True):
        self.partial_fit_(X_data, Y_data, show_loss)
        self.explorations_mab[Y_data[0]].reward(exp_factor)

    def predict(self, X_data):
        pred = self.predict_(X_data)[0]
        exp_factor = self.explorations_mab[pred].select()[0]
        if np.random.uniform() < self.e[exp_factor]:
            removed_arms = self.arms_values.clone().numpy().tolist()
            removed_arms.remove(pred)
            return random.choice(removed_arms), exp_factor

        return pred, exp_factor
