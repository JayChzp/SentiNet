import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import numpy as np
import data_helper
import models
import time
import metrics
from scipy.stats import spearmanr
from numpy.linalg import norm
from numpy import dot
def run_cnn_exp(data, embedding_matrix, token_to_idx, seed=0, weight_decay=0.0, lr=0.0001, max_len=51,
            batch_size=50,
            idx_to_label=['negative', 'neutral', 'positive'],
            embedding_freeze=True,
            embedding_normalize=True,
            obj='loss',
            measures=['loss', 'macro_f1', 'acc', 'avgrecall'],
            epoches=100,
            silent=False,
            cuda=-1):
    # set seed for reproduciable
    seed = seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        train_np_sentences, train_np_labels), batch_size=batch_size, sampler=sampler)
    dev_iter = DataLoader(data_helper.BasicDataset(
        dev_np_sentences, dev_np_labels), batch_size=batch_size)
    test_iter = DataLoader(data_helper.BasicDataset(
        test_np_sentences, test_np_labels), batch_size=batch_size)

    cnn = models.CNN(embedding_matrix, max_len,
                     embedding_freeze=embedding_freeze,
                     embedding_normalize=embedding_normalize,
                     num_classes=len(idx_to_label))
    if cuda != -1:
        cnn.cuda(cuda)

    # start training
    criterion = torch.nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(cnn.custom_params, lr=lr, weight_decay=weight_decay)
    obj_value = 0.0
    final_metrics = {'loss': 0.0, 'macro_f1': 0.0,
                     'acc': 0.0, 'avgrecall': 0.0}
    for epoch in range(epoches):
        start_time = time.time()
        cnn.train()
        train_sum_loss = 0.0
        train_count = 0
        train_predict = []
        train_gold = []
        for batch in train_iter:
            optimizer.zero_grad()
            
            sentences = batch['sentence']
            labels = batch['label']
            if cuda != -1:
                sentences = sentences.cuda()
                labels = labels.cuda()
            sentences = Variable(sentences)
            labels = Variable(labels)

            outputs = cnn(sentences)
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                train_predict.append(int(label))
            for label in labels.data:
                train_gold.append(int(label))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_sum_loss += loss.data[0]
            train_count += labels.shape[0]

        train_metrics_result = metrics.evaluation_metrics(
            train_gold, train_predict, measures=measures, idx_to_label=idx_to_label)
        train_metrics_result['loss'] = train_sum_loss / train_count
        if not silent:
            output_str = "[{}/{}]\ntrain\t".format(epoch + 1, epoches)
            for key in measures:
                output_str += "{}={:.4f}\t".format(key,
                                                   train_metrics_result[key])
            print(output_str)

        cnn.eval()
        dev_sum_loss = 0.0
        dev_count = 0
        dev_predict = []
        dev_gold = []
        for batch in dev_iter:
            optimizer.zero_grad()

            sentences = batch['sentence']
            labels = batch['label']
            if cuda != -1:
                sentences = sentences.cuda()
                labels = labels.cuda()
            sentences = Variable(sentences)
            labels = Variable(labels)

            outputs = cnn(sentences)
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                dev_predict.append(int(label))
            for label in labels.data:
                dev_gold.append(int(label))
            loss = criterion(outputs, labels)
            dev_sum_loss += loss.data[0]
            dev_count += labels.shape[0]

        dev_metrics_result = metrics.evaluation_metrics(
            dev_gold, dev_predict, measures=measures, idx_to_label=idx_to_label)
        dev_metrics_result['loss'] = dev_sum_loss / dev_count
        if not silent:
            output_str = "dev\t".format(epoch + 1, epoches)
            for key in measures:
                output_str += "{}={:.4f}\t".format(key,
                                                   dev_metrics_result[key])
            print(output_str)

        test_sum_loss = 0.0
        test_count = 0
        test_predict = []
        test_gold = []
        for batch in test_iter:
            optimizer.zero_grad()

            sentences = batch['sentence']
            labels = batch['label']
            if cuda != -1:
                sentences = sentences.cuda()
                labels = labels.cuda()
            sentences = Variable(sentences)
            labels = Variable(labels)

            outputs = cnn(sentences)
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                test_predict.append(int(label))
            for label in labels.data:
                test_gold.append(int(label))
            loss = criterion(outputs, labels)
            test_sum_loss += loss.data[0]
            test_count += labels.shape[0]

        test_metrics_result = metrics.evaluation_metrics(
            test_gold, test_predict, measures=measures, idx_to_label=idx_to_label)
        test_metrics_result['loss'] = test_sum_loss / test_count
        if not silent:
            output_str = "test\t".format(epoch + 1, epoches)
            for key in measures:
                output_str += "{}={:.4f}\t".format(key,
                                                   round(test_metrics_result[key], 5))
            print(output_str)

        # output time
        if not silent:
            print("cost time:{}".format(time.time() - start_time))

        # early stop procedure
        if epoch == 0:
            obj_value = dev_metrics_result[obj]
            final_metrics = test_metrics_result
        else:
            if obj != 'loss':
                if dev_metrics_result[obj] > obj_value:
                    obj_value = dev_metrics_result[obj]
                    final_metrics = test_metrics_result
            else:
                if dev_metrics_result[obj] < obj_value:
                    obj_value = dev_metrics_result[obj]
                    final_metrics = test_metrics_result

    return obj_value, final_metrics, cnn.embed.weight.data.cpu().numpy()

def run_rnn_exp(data, embedding_matrix, token_to_idx, seed=0, weight_decay=0.0, lr=0.001, max_len=51,
            batch_size=128,
            idx_to_label=['negative', 'neutral', 'positive'],
            embedding_freeze=True,
            embedding_normalize=True,
            obj='loss',
            measures=['loss', 'macro_f1', 'acc', 'avgrecall'],
            epoches=50,
            silent=False,
            cuda=-1):
    # set seed for reproduciable
    seed = seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data into numpy format
    train_list_sentences, train_list_labels = data_helper.map_to_num_rnn(
        data['train'][0], data['train'][1], token_to_idx, idx_to_label=idx_to_label, max_len=max_len)
    dev_list_sentences, dev_list_labels = data_helper.map_to_num_rnn(
        data['dev'][0], data['dev'][1], token_to_idx, idx_to_label=idx_to_label, max_len=max_len)
    test_list_sentences, test_list_labels = data_helper.map_to_num_rnn(
        data['test'][0], data['test'][1], token_to_idx, idx_to_label=idx_to_label, max_len=max_len)

    # create sampler to solve imbalance of training data
    train_num_count = [0] * len(idx_to_label)
    for label in train_list_labels:
        train_num_count[label] += 1
    if not silent:
        print(train_num_count)
    sample_weights = [0.0] * len(train_list_labels)
    for i, label in enumerate(train_list_labels):
        sample_weights[i] = len(train_list_labels) / train_num_count[label]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))

    # create iter for train, dev and test
    train_iter = DataLoader(data_helper.BasicDataset(
        train_list_sentences, train_list_labels), batch_size=batch_size, sampler=sampler, collate_fn=data_helper.rnn_collate_fn_cuda)
    dev_iter = DataLoader(data_helper.BasicDataset(
        dev_list_sentences, dev_list_labels), batch_size=batch_size, collate_fn=data_helper.rnn_collate_fn_cuda)
    test_iter = DataLoader(data_helper.BasicDataset(
        test_list_sentences, test_list_labels), batch_size=batch_size, collate_fn=data_helper.rnn_collate_fn_cuda)

    model = models.BiLSTM(embedding_matrix, 
                          hidden_size=150, 
                          num_layer=2, 
                          embedding_freeze=True, 
                          embedding_normalize=True, 
                          max_norm=5.0, 
                          num_classes=2)
    if cuda != -1:
        model.cuda(cuda)

    # start training
    criterion = torch.nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.custom_params, lr=lr, weight_decay=weight_decay)
    obj_value = 0.0
    final_metrics = {'loss': 0.0, 'macro_f1': 0.0,
                     'acc': 0.0, 'avgrecall': 0.0}
    for epoch in range(epoches):
        start_time = time.time()
        model.train()
        train_sum_loss = 0.0
        train_count = 0
        train_predict = []
        train_gold = []
        batch = ""
        for batch in train_iter:
            model.hidden1 = model.init_hidden(batch_size=int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size=int(batch['labels'].data.size()[0]))
            optimizer.zero_grad()
            outputs = model(batch['sentence'])
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                train_predict.append(int(label))
            for label in batch['labels'].data:
                train_gold.append(int(label))
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            train_sum_loss += loss.data[0]
            train_count += batch['labels'].shape[0]

        train_metrics_result = metrics.evaluation_metrics(
            train_gold, train_predict, measures=measures, idx_to_label=idx_to_label)
        train_metrics_result['loss'] = train_sum_loss / train_count
        if not silent:
            output_str = "[{}/{}]\ntrain\t".format(epoch + 1, epoches)
            for key in measures:
                output_str += "{}={:.4f}\t".format(key,
                                                   train_metrics_result[key])
            print(output_str)

        model.eval()
        dev_sum_loss = 0.0
        dev_count = 0
        dev_predict = []
        dev_gold = []
        for batch in dev_iter:
            model.hidden1 = model.init_hidden(batch_size=int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size=int(batch['labels'].data.size()[0]))
            optimizer.zero_grad()
            outputs = model(batch['sentence'])
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                dev_predict.append(int(label))
            for label in batch['labels'].data:
                dev_gold.append(int(label))
            loss = criterion(outputs, batch['labels'])
            dev_sum_loss += loss.data[0]
            dev_count += batch['labels'].shape[0]

        dev_metrics_result = metrics.evaluation_metrics(
            dev_gold, dev_predict, measures=measures, idx_to_label=idx_to_label)
        dev_metrics_result['loss'] = dev_sum_loss / dev_count
        if not silent:
            output_str = "dev\t".format(epoch + 1, epoches)
            for key in measures:
                output_str += "{}={:.4f}\t".format(key,
                                                   dev_metrics_result[key])
            print(output_str)

        test_sum_loss = 0.0
        test_count = 0
        test_predict = []
        test_gold = []
        for batch in test_iter:
            model.hidden1 = model.init_hidden(batch_size=int(batch['labels'].data.size()[0]))
            model.hidden2 = model.init_hidden(batch_size=int(batch['labels'].data.size()[0]))
            optimizer.zero_grad()
            outputs = model(batch['sentence'])
            _, outputs_label = torch.max(outputs, 1)
            for label in outputs_label.data:
                test_predict.append(int(label))
            for label in batch['labels'].data:
                test_gold.append(int(label))
            loss = criterion(outputs, batch['labels'])
            test_sum_loss += loss.data[0]
            test_count += batch['labels'].shape[0]

        test_metrics_result = metrics.evaluation_metrics(
            test_gold, test_predict, measures=measures, idx_to_label=idx_to_label)
        test_metrics_result['loss'] = test_sum_loss / test_count
        if not silent:
            output_str = "test\t".format(epoch + 1, epoches)
            for key in measures:
                output_str += "{}={:.4f}\t".format(key,
                                                   round(test_metrics_result[key], 5))
            print(output_str)

        # output time
        if not silent:
            print("cost time:{}".format(time.time() - start_time))

        # early stop procedure
        if epoch == 0:
            obj_value = dev_metrics_result[obj]
            final_metrics = test_metrics_result
        else:
            if obj != 'loss':
                if dev_metrics_result[obj] > obj_value:
                    obj_value = dev_metrics_result[obj]
                    final_metrics = test_metrics_result
            else:
                if dev_metrics_result[obj] < obj_value:
                    obj_value = dev_metrics_result[obj]
                    final_metrics = test_metrics_result

    return obj_value, final_metrics, model.embed.weight.data.cpu().numpy()

def distance(v1, v2, normalised_vectors=True):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))

def simlex_analysis(token_to_idx, embeddings):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    The method also prints the gold standard SimLex-999 ranking to results/simlex_ranking.txt,
    and the ranking produced using the counter-fitted vectors to results/counter_ranking.txt
    """
    fread_simlex = open("dataset/SimLex-999/SimLex-999.txt", "r")
    pair_list = []

    line_number = 0
    for line in fread_simlex:
        if line_number > 0:
            tokens = line.split('\t')
            word_i = tokens[0]
            word_j = tokens[1]
            score = float(tokens[3])
            if word_i in token_to_idx and word_j in token_to_idx:
                pair_list.append(((word_i, word_j), score))
        line_number += 1

    pair_list.sort(key=lambda x: - x[1])

    extracted_list = []
    extracted_scores = {}

    for (x, y) in pair_list:

        (word_i, word_j) = x
        current_distance = distance(
            embeddings[token_to_idx[word_i]], embeddings[token_to_idx[word_j]])
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return spearman_rho[0]