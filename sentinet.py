import embedding
import models
import argparse
import data_helper
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
import random
import numpy as np
import metrics
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import dataset
import time
import os
from my_experiment import simlex_analysis
import time
start_time = time.time()

class CNN(nn.Module):
    def __init__(self, embeddings, sentence_size,
                 filters=[3,4,5],
                 num_filters=100,
                 embedding_normalize=False,
                 max_norm=5.0,
                 num_classes=3,
                 l2_constraint=3.0,
                 senti_hidden_size=300):

        super(CNN,self).__init__()
        self.l2_constraint=l2_constraint
        self.cnn_params = []
        self.embed_params = []
        vocab_size=embeddings.shape[0]
        embed_size=embeddings.shape[1]
        self.embed_size = embed_size
        self.embed=nn.Embedding(vocab_size, embed_size, max_norm=max_norm)
        torch_wv = torch.from_numpy(embeddings)
        if embedding_normalize:
            torch_wv = torch.nn.functional.normalize(torch_wv)

        self.embed.weight=nn.Parameter(torch_wv, requires_grad=True)
        for param in self.embed.parameters():
            self.embed_params.append(param)

        self.conv_list=nn.ModuleList()
        self.sentence_size=sentence_size
        self.filters=filters
        self.num_filters=num_filters
        for filter in filters:
            conv = nn.Conv2d(1,num_filters,(filter,embed_size))
            for params in conv.parameters():
                self.cnn_params.append(params)
                if params.data.dim()>1:
                    nn.init.uniform(params.data, -0.01, 0.01)
                else:
                    params.data.fill_(0.0)
            self.conv_list.append(conv)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(3*num_filters, num_classes)
        for params in self.fc.parameters():
            if params.data.dim()>1:
                    nn.init.uniform(params.data, -0.01, 0.01)
            else:
                params.data.fill_(0.0)
            self.cnn_params.append(params)

        # constraint network
        self.senti_params = []
        self.senti_fc1 = nn.Linear(embed_size, senti_hidden_size)
        for param in self.senti_fc1.parameters():
            if params.data.dim()>1:
                    nn.init.uniform(params.data, -0.01, 0.01)
            else:
                params.data.fill_(0.0)
            self.senti_params.append(param)

        self.senti_fc2 = nn.Linear(senti_hidden_size, 3)
        for param in self.senti_fc2.parameters():
            if params.data.dim()>1:
                    nn.init.uniform(params.data, -0.01, 0.01)
            else:
                params.data.fill_(0.0)
            self.senti_params.append(param)


    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0),1,x.size(1),x.size(2))
        convs_output=[]
        for filter,conv in zip(self.filters,self.conv_list):
            conv_output = conv(x)
            pooling_output = F.max_pool2d(F.relu(conv_output),(self.sentence_size-filter+1,1))
            pooling_output = pooling_output.view(-1,self.num_filters)
            convs_output.append(pooling_output)
        output = torch.cat(convs_output,1)
        output = self.dropout(output)
        output = self.fc(output)
        return output

    def senti_forward(self, x):
        x = self.embed(x).view(-1, self.embed_size)
        output = self.senti_fc2(F.sigmoid(self.senti_fc1(x)))
        return output

    def weight_norm(self):
        for params in self.fc.parameters():
            if params.data.dim() == 2:
                l2_norm = torch.norm(params.data, p=2)
                if l2_norm > self.l2_constraint:
                    params.data /= (l2_norm / 3.0)

def run_exp(data, embedding_matrix, token_to_idx, seed=0, weight_decay=0.0, lr=0.001, max_len=51,
            batch_size=128,
            idx_to_label=['negative', 'neutral', 'positive'],
            embedding_normalize=True,
            obj='loss',
            silent=True,
            measures=['loss', 'macro_f1', 'acc', 'avgrecall'],
            epoches=50,
            embed_lr=0.0001,
            embed_weight_decay=0.01,
            start_optimize_embedding_epoch=50):
    # set seed for reproduciable
    seed = seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load senti information
    idx_to_token = [int(0)] * len(token_to_idx)
    for token in token_to_idx:
        idx_to_token[token_to_idx[token]] = token

    word_to_synsets = {}
    for i, word in enumerate(idx_to_token):
        if word == word.lower():
            word_synsets = wn.synsets(word)
            if len(word_synsets) > 0:
                most_count = -1
                represent_synset = ""
                for synset in word_synsets:
                    count = 0
                    lemmas = synset.lemmas()
                    if len(lemmas) > 0:
                        for lemma in lemmas:
                            count += lemma.count()
                        if count > most_count:
                            most_count = count
                            represent_synset = synset
                if most_count > 0:
                    word_to_synsets[word] = str(represent_synset)

    # sentiment score between [-1, 1]
    idx_to_senti_score = {}
    for word in word_to_synsets:
        str_synset = word_to_synsets[word].split('\'')[1]
        try:
            senti_synset = swn.senti_synset(str_synset)
            senti_score = senti_synset.pos_score() - senti_synset.neg_score()
            if senti_score > 0:
                idx_to_senti_score[token_to_idx[word]] = 2
            elif senti_score < 0:
                idx_to_senti_score[token_to_idx[word]] = 0
            else:
                idx_to_senti_score[token_to_idx[word]] = 1
        except ValueError as e:
            pass

    # Load data into numpy format
    train_np_sentences, train_np_labels = data_helper.map_to_num(
        data['train'][0], data['train'][1], token_to_idx, idx_to_label=idx_to_label, max_len=max_len)
    dev_np_sentences, dev_np_labels = data_helper.map_to_num(
        data['dev'][0], data['dev'][1], token_to_idx, idx_to_label=idx_to_label, max_len=max_len)
    test_np_sentences, test_np_labels = data_helper.map_to_num(
        data['test'][0], data['test'][1], token_to_idx, idx_to_label=idx_to_label, max_len=max_len)

    # create sampler to solve imbalance of training data
    train_num_count = [0] * len(idx_to_label)
    for label in train_np_labels:
        train_num_count[label] += 1
    sample_weights = [0.0] * len(train_np_labels)
    for i, label in enumerate(train_np_labels):
        sample_weights[i] = len(train_np_labels) / train_num_count[label]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))

    # create iter for train, dev and test
    train_iter = DataLoader(data_helper.BasicDataset(
        train_np_sentences, train_np_labels), batch_size=batch_size, pin_memory=True, sampler=sampler)
    dev_iter = DataLoader(data_helper.BasicDataset(
        dev_np_sentences, dev_np_labels), batch_size=batch_size, pin_memory=True)
    test_iter = DataLoader(data_helper.BasicDataset(
        test_np_sentences, test_np_labels), batch_size=batch_size, pin_memory=True)

    cnn = CNN(embedding_matrix, max_len,
                     embedding_normalize=embedding_normalize,
                     num_classes=len(idx_to_label),
                     max_norm=1.0)
    cnn.cuda()

    # start training
    criterion = torch.nn.CrossEntropyLoss()
    senti_criterion = torch.nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(cnn.cnn_params, lr=lr, weight_decay=weight_decay)
    embedding_optimizer = torch.optim.SGD(cnn.embed_params, lr=embed_lr, weight_decay=embed_weight_decay)
    senti_optimizer = torch.optim.Adam(cnn.senti_params, lr=0.001, weight_decay=weight_decay)
    obj_value = 0.0
    final_metrics = {'loss': 0.0, 'macro_f1': 0.0,
                     'acc': 0.0, 'avgrecall': 0.0}
    for epoch in range(0, epoches):
        if epoch == start_optimize_embedding_epoch:
            cnn.load_state_dict(torch.load("checkpoint/checkpoint.ckp"))
        start_time = time.time()
        cnn.train()
        train_sum_loss = 0.0
        senti_train_sum_loss = 0.0
        train_count = 0
        senti_train_count = 0
        train_predict = []
        train_gold = []
        for batch in train_iter:
            embedding_optimizer.zero_grad()
            cnn_optimizer.zero_grad()
            sentences = Variable(batch['sentence'].cuda())
            labels = Variable(batch['label'].cuda())
            outputs = cnn(sentences)
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                train_predict.append(int(label))
            for label in labels.data:
                train_gold.append(int(label))
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)

            train_sum_loss += loss.data[0] * labels.shape[0]
            train_count += labels.shape[0]

            # trainig senti net
            senti_optimizer.zero_grad()
            pos_words = []
            neg_words = []
            neutral_words = []
            for sentence in batch['sentence']:
                for idx in sentence:
                    if idx in idx_to_senti_score:
                        if idx_to_senti_score[idx] == 0:
                            neg_words.append(idx)
                        elif idx_to_senti_score[idx] == 1:
                            neutral_words.append(idx)
                        else:
                            pos_words.append(idx)
            words = []
            words_label = []
            for i in range(50):
                ran_num = random.random()
                if ran_num < 0.33:
                    idx = random.choice(pos_words)
                    words.append(idx)
                    words_label.append(idx_to_senti_score[idx])
                elif ran_num < 0.67:
                    idx = random.choice(neg_words)
                    words.append(idx)
                    words_label.append(idx_to_senti_score[idx])
                else:
                    idx = random.choice(neutral_words)
                    words.append(idx)
                    words_label.append(idx_to_senti_score[idx])

            words = Variable(torch.cuda.LongTensor(words))
            words_label = Variable(torch.cuda.LongTensor(words_label))
            output = cnn.senti_forward(words)
            senti_loss = senti_criterion(output, words_label)
            senti_loss.backward()
            senti_train_sum_loss += senti_loss.data[0] * words_label.shape[0]
            senti_train_count += words_label.shape[0]

            # BEFORE-Senti
            #if epoch < start_optimize_embedding_epoch:
            #    embedding_optimizer.step()
            #    senti_optimizer.step()
            #else:
            #    cnn_optimizer.step()

            # DURING-Senti
            #cnn_optimizer.step()
            #senti_optimizer.step()
            #embedding_optimizer.step()

            # AFTER-Senti
            if epoch < start_optimize_embedding_epoch:
                cnn_optimizer.step()
                senti_optimizer.step()
            else:
                embedding_optimizer.step()
            cnn.weight_norm()


        cnn.eval()
        dev_sum_loss = 0.0
        dev_count = 0
        dev_predict = []
        dev_gold = []
        for batch in dev_iter:
            sentences = Variable(batch['sentence'].cuda())
            labels = Variable(batch['label'].cuda())
            outputs = cnn(sentences)
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                dev_predict.append(int(label))
            for label in labels.data:
                dev_gold.append(int(label))
            loss = criterion(outputs, labels)
            dev_sum_loss += loss.data[0] * labels.shape[0]
            dev_count += labels.shape[0]

        dev_metrics_result = metrics.evaluation_metrics(
            dev_gold, dev_predict, measures=measures, idx_to_label=idx_to_label)
        dev_metrics_result['loss'] = dev_sum_loss / dev_count


        test_sum_loss = 0.0
        test_count = 0
        test_predict = []
        test_gold = []
        for batch in test_iter:
            sentences = Variable(batch['sentence'].cuda())
            labels = Variable(batch['label'].cuda())
            outputs = cnn(sentences)
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                test_predict.append(int(label))
            for label in labels.data:
                test_gold.append(int(label))
            loss = criterion(outputs, labels)
            test_sum_loss += loss.data[0] * labels.shape[0]
            test_count += labels.shape[0]

        test_metrics_result = metrics.evaluation_metrics(
            test_gold, test_predict, measures=measures, idx_to_label=idx_to_label)
        test_metrics_result['loss'] = test_sum_loss / test_count

        # stop procedure
        os.makedirs("checkpoint", exist_ok=True)
        if epoch == 0:
            torch.save(cnn.state_dict(), "checkpoint/checkpoint.ckp")
            obj_value = dev_metrics_result[obj]
            final_metrics = test_metrics_result
        else:
            if obj != 'loss':
                if dev_metrics_result[obj] > obj_value:
                    torch.save(cnn.state_dict(), "checkpoint/checkpoint.ckp")
                    obj_value = dev_metrics_result[obj]
                    final_metrics = test_metrics_result
            else:
                if dev_metrics_result[obj] < obj_value:
                    torch.save(cnn.state_dict(), "checkpoint/checkpoint.ckp")
                    obj_value = dev_metrics_result[obj]
                    final_metrics = test_metrics_result

    return obj_value, final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_len', help="define the max length of sentence", type=int, default=51)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument("--embedding_freeze",
                        help="freeze embedding", action="store_true")
    parser.add_argument("--embedding_normalize",
                        help="whether to normalize word embedding", action="store_true")
    parser.add_argument("--obj", default="macro_f1")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_path", default="embedding/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--isBinary", help="indicate binary format of embedding", action="store_true")
    parser.add_argument("--semeval2016", help="run SemEval2016", action="store_true")
    parser.add_argument("--semeval2017", help="run SemEval2017", action="store_true")
    parser.add_argument("--mr", help="run MR test", action="store_true")
    parser.add_argument("--sst2", help="run SST2", action="store_true")
    parser.add_argument("--all", help="run all test", action="store_true")
    args = parser.parse_args()

    # run the test for SemEval2016
    if args.semeval2016 or args.all:
        lr = 1e-3
        weight_decay = 0
        embed_lr = 1e-2
        embed_weight_decay = 1e-4

        data = dataset.load.SemEval(year="2016", leave_dev=True, fraction=0.1, seed=args.seed)
        print("number of train data:{}".format(len(data['train'][0])))
        print("number of dev data:{}".format(len(data['dev'][0])))
        print("number of test data:{}".format(len(data['test'][0])))
        word_list = dataset.load.get_word_list(data['train'][0])
        embedding_matrix, token_to_idx, idx_to_token = embedding.load_embedding(args.embedding_path, word_list, isBinary=args.isBinary)

        obj_value, result = run_exp(data, embedding_matrix, token_to_idx, seed=args.seed, weight_decay=weight_decay,
                                lr=lr, max_len=args.max_len, batch_size=50,
                                obj='loss', embed_lr=embed_lr, epoches=100,
                                embed_weight_decay=embed_weight_decay)

        print("{}\t{}\t{}".format(obj_value, result['macro_f1'], embed_weight_decay))
        print("SemEval2016 Acc={:.4f} Macro F1={:.4f} AvgRecall={:.4f}".format(
            result['acc'], result['macro_f1'], result['avgrecall']))

    # run the test for SemEval2017
    if args.semeval2017 or args.all:
        lr = 1e-3
        weight_decay = 0
        embed_lr = 1e-2
        embed_weight_decay = 1e-4

        data = dataset.load.SemEval(year="2017", leave_dev=True, fraction=0.1, seed=args.seed)
        print("number of train data:{}".format(len(data['train'][0])))
        print("number of dev data:{}".format(len(data['dev'][0])))
        print("number of test data:{}".format(len(data['test'][0])))
        word_list = dataset.load.get_word_list(data['train'][0])
        embedding_matrix, token_to_idx, idx_to_token = embedding.load_embedding(args.embedding_path, word_list, isBinary=args.isBinary)

        obj_value, result = run_exp(data, embedding_matrix, token_to_idx, seed=args.seed, weight_decay=weight_decay,
                                lr=lr, max_len=args.max_len, batch_size=50,
                                obj='loss', embed_lr=embed_lr, epoches=100,
                                embed_weight_decay=embed_weight_decay)

        print("{}\t{}\t{}".format(obj_value, result['macro_f1'], embed_weight_decay))
        print("SemEval2017 Acc={:.4f} Macro F1={:.4f} AvgRecall={:.4f}".format(
            result['acc'], result['macro_f1'], result['avgrecall']))

    # run the test for Movie Review
    if args.mr or args.all:
        lr = 1e-3
        weight_decay = 0
        embed_lr = 1e-2
        embed_weight_decay = 1e-4

        sentences, labels = dataset.load.MR()
        word_list = dataset.load.get_word_list(sentences)
        embedding_matrix, token_to_idx, idx_to_token = embedding.load_embedding(args.embedding_path, word_list, isBinary=args.isBinary)
        cv_helper = dataset.load.CrossValidationHelper(sentences, labels, cv=10, seed=args.seed)
        acc_result = np.zeros([10], dtype=np.float32)
        for cv_idx in range(10):
            data = cv_helper.CV(cv_idx)
            obj_value, result = run_exp(data, embedding_matrix, token_to_idx, seed=args.seed, weight_decay=weight_decay, lr=lr, max_len=args.max_len,
                            batch_size=50, obj='loss', epoches=100, silent=True, idx_to_label=['negative', 'positive'], embed_lr=embed_lr, embed_weight_decay=embed_weight_decay)
            print("MR CV_{} Acc={:.4f}".format(cv_idx + 1, result['acc']))
            acc_result[cv_idx] = result['acc']
        print('Movie Review Acc={:.4f}'.format(np.average(acc_result)))

    # run the test for SST2
    if args.sst2 or args.all:
        lr = 1e-3
        weight_decay = 0
        embed_lr = 1e-2
        embed_weight_decay = 1e-4
        data = dataset.load.SST2()

        word_list = dataset.load.get_word_list(data['train'][0])
        embedding_matrix, token_to_idx, idx_to_token = embedding.load_embedding(args.embedding_path, word_list, isBinary=args.isBinary)

        obj_value, result = run_exp(data, embedding_matrix, token_to_idx, seed=args.seed, weight_decay=weight_decay,
                        lr=lr, max_len=args.max_len, batch_size=50, obj='loss', epoches=100,
                        idx_to_label=['negative', 'positive'], embed_lr=embed_lr, embed_weight_decay=embed_weight_decay)

        print("SST2 Acc={:.4f} Macro F1={:.4f} AvgRecall={:.4f}".format(
            result['acc'], result['macro_f1'], result['avgrecall']))

    print("Cost Time = {:2f}s".format(time.time()-start_time))
