import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import sys
from collections import defaultdict
#from torchnlp.word_to_vector import FastText

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self,x):
        mp = nn.MaxPool1d(1,x.size(2))
        return mp(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self,x):
        return x.squeeze()

class textCNN(torch.nn.Module):
    def __init__(self,nwords,emb_size,num_filters,windows,ntags,w2i):
        super(textCNN,self).__init__()
        
        """layers"""
        #TODO change to fastText embedding
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        #torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        #pickleFile = open("fasttext/fastText_",fastText_embeddings.pkl)

        self.ntags = ntags
        embedding_matrix = np.zeros((nwords,emb_size))
        fname = "fasttext/crawl-300d-2M-subword.vec"
        vectors = self.load_vectors(fname)
        unk = np.random.randn(1,emb_size)
        for w,i in w2i.items():
            try:
                embedding_matrix[i] = vectors[w]
            except KeyError:
                embedding_matrix[i] = unk
        self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())

        self.conv1 = nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=5)
        #self.bn = nn.BatchNorm1d(num_filters)
        self.maxpool1 = nn.MaxPool1d(5)


        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, padding=1)
        self.maxpool2 = nn.MaxPool1d(5)

        #self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5)
        #self.maxpool3 = nn.MaxPool1d(5)

        self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        #self.cnns = nn.ModuleList()
        #cnns = []
        #TODO run different CNNs (each filter_size) in parallel
        #for window_size in windows:
        #    #self.cnns.append(nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
        #    #                           stride=1, padding=0, dilation=1, groups=1, bias=True))
        #    
        #    # Conv 1d
        #    conv = nn.Sequential(nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
        #                               stride=1, padding=0, dilation=1, groups=1, bias=True),
        #                               nn.BatchNorm1d(num_filters),
        #                               #PrintLayer(),
        #                               MaxPool(),
        #                               #PrintLayer(),
        #                               nn.ReLU(),
        #                               #PrintLayer(),
        #                               Flatten())
        #                               #,PrintLayer())
        #    cnns.append(conv)
        #    
        #self.cnns = nn.ModuleList(cnns)
         
        #TODO add dropout
        self.dropout = nn.Dropout(p=0.5)
        #TODO add L2 norm

    def forward(self, words, return_activations=False):
        #print (words.size())
        emb = self.embedding(words)                 # batch_size x nwords x emb_size
        #print (emb.size())
        print (emb.size())
        emb = emb.permute(0,2,1)       # 1 x emb_size x nwords
        print (emb.size())
       
        print ("conv") 
        op  = self.conv1(emb)

        #op  = self.bn(op)
        print (op.size())
        op  = F.relu(op)
        print ("pool") 
        op  = self.maxpool1(op)
        print (op.size())

        print ("conv") 
        op  = self.conv2(op)    
        print (op.size())
        op  = F.relu(op)
        print ("pool") 
        op  = self.maxpool2(op)
        print (op.size())

        #print ("conv") 
        #op  = self.conv3(op)    
        #print (op.size())
        #op  = F.relu(op)
        #print ("pool") 
        #op  = self.maxpool3(op)
        #print (op.size())

        
        #print (emb.size())
        #ops = [cnn(emb) for cnn in self.cnns]
        #outputs = torch.cat(ops, dim=1)
        #print (outputs.size())
        op  = self.dropout(op)

        op  = op.permute(0,2,1)
        
        output  = self.projection_layer(op)
        print (output.size())
        sys.exit(0)
        return output
        """
        h = self.conv_1d(emb)                       # 1 x num_filters x nwords
        activations = h.squeeze().max(dim=0)[1]     # argmax along length of the sentence
        # Do max pooling
        h = h.max(dim=2)[0]                         # 1 x num_filters
        h = self.relu(h)
        features = h.squeeze()
        out = self.projection_layer(h)              # size(out) = 1 x ntags
        
        if return_activations:
            return out, activations.data.cpu().numpy(), features.data.cpu().numpy()
        return out        
        """
    def load_vectors(self,fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:])#map(float, tokens[1:])
        return data
