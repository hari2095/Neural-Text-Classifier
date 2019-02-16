import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
from collections import defaultdict
from model import *
from data_loader import *
import csv

PATH = "saved_models/best_model.pkl"
EMBEDDINGS_FILE = "fasttext/crawl-300d-2M-subword.vec"
def to_tensor(arr):
    # list -> Tensor (on GPU if possible)
    if torch.cuda.is_available():
        tensor = torch.tensor(arr).type(torch.cuda.LongTensor)
    else:
        tensor = torch.tensor(arr).type(torch.LongTensor)
    return tensor

def predict(model, test_loader, op_file, i2t):
    with torch.no_grad():
        model.eval()
        model.to(device)

        i = 0
        op_f = open(op_file,'w')
        lines = list(list())
        #header = ['label']
        writer = csv.writer(op_f)
        #lines.append(header)
        for sents,lens,labels in test_loader:
            data = to_tensor(sents)
            outputs = model(data)
            op = torch.argmax(outputs,1)
            for o in op:
                lines.append([i2t[o.item()]])
        writer.writerows(lines)
batch_size = 50

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
FILTER_SIZE = 100

s = 3

# initialize the model
model = textCNN(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZES, ntags,dl.w2i,EMBEDDINGS_FILE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    print ("cuda available")
    type = torch.cuda.LongTensor
    model.cuda()

eps = 1e-4
#for t in trainX:
#    print (t.size())
prev_dev_loss = float('inf')
dev_loss = 0
prev_train_loss = float('inf')
train_loss = 0
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

fall_cnt = 0

for ITER in range(100):
    # Perform training
    #random.shuffle(train)
    model.train()
    model.cuda()
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
            
    print("iter %r: train loss/sent=%.4f, train acc=%.4f, time=%.2fs" % (ITER, train_loss/ntrain, train_correct/ntrain, time.time()-start))
    # Perform testing
    model.eval()
    model.to(device)
    dev_correct = 0.0
    dev_loss = 0
    start = time.time()
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
        dev_correct += torch.sum(preds.eq(y_batch)).item() 
        my_loss = criterion(scores,y_batch)
        dev_loss += my_loss.item()
        #predict = scores[0].argmax().item()
        #if predict == tag:
        #    test_correct += 1
    print("iter %r: dev loss/sent=%.4f, dev acc=%.4f, time=%.2fs" % (ITER, dev_loss/ndev, dev_correct/ndev, time.time()-start))
    #If dev loss has increased, break, else save model and check if it has converged
    if dev_loss > prev_dev_loss :
        print ("Not saving model")
        break
    torch.save(model.state_dict(), PATH)
    print ("Saved model's dict to disk\n")
    prev_dev_loss = dev_loss
    if (prev_train_loss - train_loss <= eps):
        print ("Model has converged")
        break
    prev_train_loss = train_loss

t2i = dl.t2i
i2t = {}
for t,i in t2i.items():
    i2t[i] = t
model.load_state_dict(torch.load(PATH))
op_file = "topicclass_valid.labels"
predict(model, dl.dev_data_loader, op_file, i2t)
op_file = "topicclass_test.labels"
predict(model, dl.test_data_loader, op_file, i2t)
