import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
from collections import defaultdict
from model2 import *
from data_loader import *
"""
# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield (words,[w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
nwords = len(w2i)
ntags = len(t2i)

print (ntags)
"""
def to_tensor(arr):
    # list -> Tensor (on GPU if possible)
    if torch.cuda.is_available():
        tensor = torch.tensor(arr).type(torch.cuda.LongTensor)
    else:
        tensor = torch.tensor(arr).type(torch.LongTensor)
    return tensor

batch_size = 64

dl = DataLoader(batch_size)
ntrain = dl.num_samples_train
ndev   = dl.num_samples_dev

nwords = dl.nwords
ntags  = dl.ntags
# Define the model
EMB_SIZE = 300
WIN_SIZE1 = 3
WIN_SIZE2 = 4
WIN_SIZE3 = 5
WIN_SIZES = [WIN_SIZE1,WIN_SIZE2,WIN_SIZE3]
FILTER_SIZE = 128

s = 3

# initialize the model
model = textCNN(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZES, ntags,dl.w2i)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

for ITER in range(100):
    # Perform training
    #random.shuffle(train)
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for sents, lens, labels in dl.train_data_loader:
        # Padding (can be done in the conv layer as well)
        #if len(wids) < WIN_SIZE3:
        #    wids += [0] * (WIN_SIZE3 - len(wids))
        #words_tensor = torch.tensor(wids).type(type)
        #tag_tensor = torch.tensor([tag]).type(type)
        #scores = model(words_tensor)
        #predict = scores[0].argmax().item()
        #if predict == tag:
        #    train_correct += 1
        #tag_tensor = torch.zeros(ntags)
        #tag_tensor[tag] = 1
        #print (scores.size(),tag_tensor.size())
        
        x_batch = to_tensor(sents)
        y_batch = to_tensor(labels)
        
        #TODO call forward method for the batched data
        scores = model(x_batch,lens) 
        #print (scores.size(),y_batch.size())       
        
        preds = torch.argmax(scores,dim=1)
        #print (preds.size())
        #print (torch.sum(preds.eq(y_batch)).item())
        train_correct += torch.sum(preds.eq(y_batch)).item() 
        #print (train_correct)
        #print (scores.size())
        my_loss = criterion(scores, y_batch)
        train_loss += my_loss.item()
        #l2 = 0
        #for p in model.parameters():
        #    l2 = l2 + (p**2).sum()
        #print(l2.item())
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        norm = torch.norm(model.projection_layer.weight.data).item()
        #As in Yoon Kim's paper, if L2 norm of weight exceeds a threshold s, scale it down
        if (norm > s):
            scaling_factor = norm/s
            model.projection_layer.weight.data /= scaling_factor
            
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (ITER, train_loss/ntrain, train_correct/ntrain, time.time()-start))
    # Perform testing
    model.eval()
    test_correct = 0.0
    for sents, lens, labels in dl.dev_data_loader:
        # Padding (can be done in the conv layer as well)
        #if len(wids) < WIN_SIZE3:
        #    wids += [0] * (WIN_SIZE3 - len(wids))
        #words_tensor = torch.tensor(wids).type(type)

        x_batch = to_tensor(sents)
        y_batch = to_tensor(labels)

        scores = model(x_batch,lens)
        #print (scores.size(), y_batch.size())
        preds = torch.argmax(scores,dim=1)
        #print (torch.sum(preds.eq(y_batch)).item())
        test_correct += torch.sum(preds.eq(y_batch)).item() 
        #predict = scores[0].argmax().item()
        #if predict == tag:
        #    test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct/ndev))
