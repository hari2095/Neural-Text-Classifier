import numpy as np
import torch
import torch.utils.data
from collections import defaultdict
from nltk.stem import *

class DataLoader:
    def __init__(self, batch_size):

        # Dictionaries for word to index and vice versa
        w2i = defaultdict(lambda: len(w2i))
        t2i = defaultdict(lambda: len(t2i))
        # Adding unk token
        UNK = w2i["<unk>"]

        # Read in the data and store the dicts
        small_train = "../nn4nlp-code/data/classes/train.txt"
        small_dev   = "../nn4nlp-code/data/classes/dev.txt"

        big_train   = "topicclass/topicclass_train.txt"
        big_dev     = "topicclass/topicclass_valid.txt" 
        big_test    = "topicclass/topicclass_test.txt"

        self.train = list(self.read_dataset(big_train, w2i, t2i))
        w2i = defaultdict(lambda: UNK, w2i)
        self.dev = list(self.read_dataset(big_dev, w2i, t2i))
        self.test = list(self.read_dataset(big_test, w2i, t2i))
        self.t2i = t2i
        self.w2i = w2i
        self.nwords = len(w2i)
        self.ntags = len(t2i)

        # Setting pin memory and number of workers
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        # Creating data loaders
        dataset_train = ClassificationDataSet(self.train)
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                             collate_fn=dataset_train.collate, shuffle=True, **kwargs)

        dataset_dev = ClassificationDataSet(self.dev)
        self.dev_data_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size,
                                                           collate_fn=dataset_dev.collate, shuffle=False, **kwargs)
        
        dataset_test = ClassificationDataSet(self.test)
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                           collate_fn=dataset_test.collate, shuffle=False, **kwargs)
        
        self.num_samples_train = dataset_train.num_of_samples
        self.num_samples_dev   = dataset_dev.num_of_samples
        self.num_samples_test  = dataset_test.num_of_samples

    @staticmethod
    def read_dataset(filename, w2i, t2i):
        with open(filename, "r", encoding = "utf-8") as f:
            for line in f:
                tag, words = line.lower().strip().split(" ||| ")
                yield ([w2i[x.lower()] for x in words.split(" ")], t2i[tag])


class ClassificationDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data):
        super(ClassificationDataSet, self).__init__()
        # data is a list of tuples (sent, label)
        self.sents = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.num_of_samples = len(self.sents)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return self.sents[idx], len(self.sents[idx]), self.labels[idx]

    @staticmethod
    def collate(batch):
        sents = np.array([x[0] for x in batch])
        sent_lens = np.array([x[1] for x in batch])
        labels = np.array([x[2] for x in batch])

        # List of indices according to decreasing order of sentence lengths
        sorted_indices = sent_lens.argsort()[::-1]#np.flipud(np.argsort(sent_lens))
        # Sorting the elements od the batch in decreasing length order
        input_lens = sent_lens[sorted_indices]
        sents = sents[sorted_indices]
        labels = labels[sorted_indices]

        # Creating padded sentences
        sent_max_len = input_lens[0]
        padded_sents = np.zeros((len(batch), sent_max_len))
        for i, sent in enumerate(sents):
            padded_sents[i, :len(sent)] = sent

        return padded_sents, input_lens, labels
