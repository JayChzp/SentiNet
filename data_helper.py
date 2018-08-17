#!/bin/bash python
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

def map_to_num(sentences, labels, token_to_idx, 
               idx_to_label=['negative', 'neutral', 'positive'],
               max_len=56,
               padding_token="<PAD>", 
               unk_token="<UNK>"):

    label_mapper = {}
    for i, label in enumerate(idx_to_label):
        label_mapper[label] = i
    
    np_labels = np.zeros([len(labels)], dtype=np.int64)
    for i, label in enumerate(labels):
        np_labels[i] = label_mapper[label]
    
    np_sentences = np.ones([len(labels), max_len], dtype=np.int64) * token_to_idx[padding_token]
    for sen_idx, sentence in enumerate(sentences):
        for token_idx, token in enumerate(sentence):
            if token_idx < max_len:
                if token in token_to_idx:
                    np_sentences[sen_idx][token_idx] = token_to_idx[token]
                else:
                    np_sentences[sen_idx][token_idx] = token_to_idx[unk_token]
    
    return np_sentences, np_labels

def map_to_num_rnn(sentences, labels, token_to_idx, 
               idx_to_label=['negative', 'neutral', 'positive'],
               max_len=56,
               padding_token="<PAD>", 
               unk_token="<UNK>"):

    label_mapper = {}
    for i, label in enumerate(idx_to_label):
        label_mapper[label] = i
    
    list_labels = [int(0)] * len(labels)
    for i, label in enumerate(labels):
        list_labels[i] = label_mapper[label]
    
    list_sentences = []
    for sen_idx, sentence in enumerate(sentences):
        num_sentence = []
        for token_idx, token in enumerate(sentence):
            if token_idx < max_len:
                if token in token_to_idx:
                    num_sentence.append(token_to_idx[token])
                else:
                    num_sentence.append(token_to_idx[unk_token])
        list_sentences.append(num_sentence)
    return list_sentences, list_labels

class BasicDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return {'sentence':self.x[idx],'label':self.y[idx]}

class JointLearningDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'sentence':self.x[idx],'label':self.y[idx]}

def rnn_collate_fn_cuda(x):
    lengths = np.array([len(term['sentence']) for term in x])
    sorted_index = np.argsort(-lengths)

    # build reverse index map to reconstruct the original order
    reverse_sorted_index = np.zeros(len(sorted_index), dtype=int)
    for i, j in enumerate(sorted_index):
        reverse_sorted_index[j]=i
    lengths = lengths[sorted_index]
    
    # control the maximum length of LSTM
    max_len = lengths[0]
    batch_size = len(x)
    sentence_tensor = torch.LongTensor(batch_size, int(max_len)).zero_()
    for i, index in enumerate(sorted_index):
        sentence_tensor[i][:lengths[i]] = torch.LongTensor(x[index]['sentence'])
    labels = Variable(torch.LongTensor([x[i]['label'] for i in sorted_index]))
    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(Variable(sentence_tensor.t()).cuda(), lengths)
    return {'sentence':packed_sequences, 'labels':labels.cuda(), 'reverse_sorted_index':reverse_sorted_index}