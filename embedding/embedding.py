#!/bin/bash python
#coding:utf-8
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

def load_embedding(file_path, word_list, isBinary=True, padding_token="<PAD>", unk_token="<UNK>", seed=0):
    token_to_idx = {}
    idx_to_token = []
    word_set = set(word_list)
    wv = KeyedVectors.load_word2vec_format(file_path, binary=isBinary)

    if len(word_list) == 0:
        # load all word in word vectors
        for token in wv.vocab:
            word_set.add(token.lower())
    word_set.add(padding_token)
    word_set.add(unk_token)

    idx_to_token = list(word_set)
    for i, token in enumerate(idx_to_token):
        token_to_idx[token] = i
    
    np.random.seed(seed)
    embed_size = wv['good'].shape[0]
    embeddings = np.array(np.random.normal(0, 1, [len(idx_to_token), embed_size]), dtype=np.float32)
    print("embedding size = {} x {}".format(len(idx_to_token), embed_size))
    
    for idx, token in enumerate(idx_to_token):
        if token in wv.vocab:
            embeddings[idx] = np.array(wv[token])

    return embeddings, token_to_idx, idx_to_token

def export_embedding(file_path, embedding_matrix, idx_to_token):
    with open(file_path, "w") as f:
        f.write("{} {}\n".format(embedding_matrix.shape[0], embedding_matrix.shape[1]))
        for idx, token in enumerate(idx_to_token):
            f.write(token)
            for value in embedding_matrix[idx]:
                f.write(" {}".format(value))
            f.write("\n")