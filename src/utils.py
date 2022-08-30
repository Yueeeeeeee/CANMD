import os
from pathlib import Path
from config import args

import json
import string
import pickle
import random
from abc import *
import numpy as np
import pandas as pd

import torch.nn.functional as F
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config import args
from dataset import *
from dataloader import *

from dataloader import preprocess, tokenize, get_loader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler

from abstention.calibration import VectorScaling, TempScaling, NoBiasVectorScaling
from abstention.calibration import *


class PlattScaling(CalibratorFactory):

    def __init__(self, verbose=True):
        self.verbose=verbose

    def __call__(self, valid_preacts, valid_labels):

        lr = LR()                                                       
        #LR needs X to be 2-dimensional
        lr.fit(valid_preacts, valid_labels) 
   
        if (self.verbose): 
            print("Platt scaling coef:", lr.coef_[0][0],
                  "; intercept:",lr.intercept_[0])
    
        def calibration_func(preact):
            return lr.predict_proba(preact)
    
        return calibration_func


class IsotonicRegression(CalibratorFactory):

    def __init__(self, verbose=True):
        self.verbose = verbose 

    def __call__(self, valid_preacts, valid_labels):
        ir = IR()
        valid_preacts = valid_preacts[:, 1]
        min_valid_preact = np.min(valid_preacts)
        max_valid_preact = np.max(valid_preacts)
        assert len(valid_preacts)==len(valid_labels)
        #sorting to be safe...I think weird results can happen when unsorted
        sorted_valid_preacts, sorted_valid_labels = zip(
            *sorted(zip(valid_preacts, valid_labels), key=lambda x: x[0]))
        y = ir.fit_transform(sorted_valid_preacts, sorted_valid_labels)
    
        def calibration_func(preact):
            preact = preact[:, 1]
            preact = np.minimum(preact, max_valid_preact)
            preact = np.maximum(preact, min_valid_preact)
            predict_probs = np.zeros((len(preact), 2))
            predict_probs[:, 1] = ir.transform(preact.flatten())
            predict_probs[:, 0] = 1 - predict_probs[:, 1]
            return predict_probs

        return calibration_func


def get_psuedolabels_tfidf(
    source_types, 
    source_paths, 
    target_types, 
    target_paths, 
    conf_threshold=0.6):
    
    print('***** Start pseudo labeling (TFIDF) *****')
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    test_texts = []

    for t, p in zip(source_types, source_paths):
        dataset = get_dataset(args, 'train', t, p)  # validate pseudo-labels with val data
        data, labels = dataset.load_dataset()
        data = preprocess(args, data)
        for d, l in zip(data, labels):
            train_texts.append(d)
            train_labels.append(l)
    
    for t, p in zip(target_types, target_paths):
        dataset = get_dataset(args, 'val', t, p)  # validate pseudo-labels with val data
        data, labels = dataset.load_dataset()
        data = preprocess(args, data)
        for d, l in zip(data, labels):
            val_texts.append(d)
            val_labels.append(l)
        
        dataset = get_dataset(args, 'train', t, p)  # to-be-labeled target training data
        data, _ = dataset.load_dataset()            # we call it test_texts here
        data = preprocess(args, data)
        for d in data:
            test_texts.append(d)

    best_acc = 0
    best_f1 = 0
    best_probs = None
    for size in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=size)
        X_all = vectorizer.fit_transform(train_texts+val_texts)
        X_train = X_all[:len(train_texts)]
        X_val = X_all[len(train_texts):]
        X_test = vectorizer.fit_transform(test_texts)

        for j in range(2):
            if j == 0:
                clf = LogisticRegression().fit(X_train, train_labels)
            elif j == 1:
                clf = LogisticRegression(class_weight='balanced').fit(X_train, train_labels)
            
            y_probs = clf.predict_proba(X_val)
            # indices = y_probs.max(-1) >= conf_threshold
            # acc = accuracy_score(np.array(val_labels)[indices], y_probs.argmax(-1)[indices])
            # f1 = f1_score(np.array(val_labels)[indices], y_probs.argmax(-1)[indices])
            acc = accuracy_score(np.array(val_labels), y_probs.argmax(-1))
            f1 = f1_score(np.array(val_labels), y_probs.argmax(-1))
            n_pos = (y_probs.argmax(-1) == 1).sum()
            n_neg = (y_probs.argmax(-1) == 0).sum()
            if n_pos < 10 or n_neg < 10:
                continue
            
            if acc >= best_acc and f1 >= best_f1:
                best_acc = acc
                best_f1 = f1
                best_probs = clf.predict_proba(X_test)

    indices = best_probs.max(-1) >= conf_threshold
    test_texts = np.array(test_texts)[indices]
    pseudolabels = best_probs.argmax(-1)[indices]

    print('***** Pseudo labeling (TFIDF) with val acc of {:.3f} and f1 of {:.3f}. *****'.format(
        best_acc, best_f1
    ))
    print('***** Output {} samples, {} positive samples and {} negative samples. *****'.format(
        len(pseudolabels), (pseudolabels == 1).sum(), (pseudolabels == 0).sum()
    ))
    return test_texts, pseudolabels


def get_psuedolabels_model(
    args,
    tokenizer,
    model,
    target_types, 
    target_paths, 
    conf_threshold=0.6):
    
    print('***** Start pseudo labeling without correction *****')
    val_texts = []
    val_labels = []
    test_texts = []
    
    for t, p in zip(target_types, target_paths):
        dataset = get_dataset(args, 'val', t, p)  # validate pseudo-labels with val data
        data, labels = dataset.load_dataset()
        val_texts += preprocess(args, data).tolist()
        val_labels += labels.tolist()

        dataset = get_dataset(args, 'train', t, p)  # to-be-labeled target training data
        data, _ = dataset.load_dataset()            # we call it test_texts here
        test_texts += preprocess(args, data).tolist()

    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, val_texts, tokenizer)
    if 'roberta' in args.lm_model:
        val_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask)
    else:
        val_dataset = torch.utils.data.TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        sampler=SequentialSampler(val_dataset), batch_size=args.eval_batchsize)
    
    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, test_texts, tokenizer)
    if 'roberta' in args.lm_model:
        test_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask)
    else:
        test_dataset = torch.utils.data.TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
        sampler=SequentialSampler(test_dataset), batch_size=args.eval_batchsize)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    val_preds = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            if 'roberta' in args.lm_model:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
            else:
                input_ids, token_type_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            val_preds += outputs.logits.argmax(-1).tolist()

    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    
    best_probs = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            if 'roberta' in args.lm_model:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
            else:
                input_ids, token_type_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            best_probs += torch.softmax(outputs.logits, -1).tolist()

    indices = np.array(best_probs).max(-1) >= conf_threshold
    test_texts = np.array(test_texts)[indices]
    pseudolabels = np.array(best_probs).argmax(-1)[indices]
    
    print('***** Output {} samples, {} positive samples and {} negative samples. *****'.format(
        len(pseudolabels), (pseudolabels == 1).sum(), (pseudolabels == 0).sum()
    ))
    return test_texts, pseudolabels


def get_corrected_psuedolabels_model(
    args,
    tokenizer,
    model,
    target_types, 
    target_paths,
    conf_threshold=0.6,
    max_imbalance_multiplier=20):  # choose max_imbalance_multiplier from 5 to 20

    def probs_not_usable(corrected_probs, val_labels):
        indices = np.array(corrected_probs).max(-1) >= conf_threshold
        pseudolabels = np.array(corrected_probs).argmax(-1)[indices]
        if len(pseudolabels) == 0:
            return True
        val_ratio = (np.array(val_labels) == 0).sum() / (np.array(val_labels) == 1).sum()
        return (pseudolabels == 0).sum() / (pseudolabels == 1).sum() > max_imbalance_multiplier * val_ratio or \
            (pseudolabels == 0).sum() / (pseudolabels == 1).sum() < val_ratio / max_imbalance_multiplier

    print('***** Start pseudo labeling with correction *****')
    val_texts = []
    val_labels = []
    test_texts = []
    
    for t, p in zip(target_types, target_paths):
        dataset = get_dataset(args, 'val', t, p)  # validate pseudo-labels with val data
        data, labels = dataset.load_dataset()
        val_texts += preprocess(args, data).tolist()
        val_labels += labels.tolist()

        dataset = get_dataset(args, 'train', t, p)  # to-be-labeled target training data
        data, _ = dataset.load_dataset()            # we call it test_texts here
        test_texts += preprocess(args, data).tolist()

    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, val_texts, tokenizer)
    if 'roberta' in args.lm_model:
        val_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask)
    else:
        val_dataset = torch.utils.data.TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        sampler=SequentialSampler(val_dataset), batch_size=args.eval_batchsize)
    
    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, test_texts, tokenizer)
    if 'roberta' in args.lm_model:
        test_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask)
    else:
        test_dataset = torch.utils.data.TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
        sampler=SequentialSampler(test_dataset), batch_size=args.eval_batchsize)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    val_probs = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            if 'roberta' in args.lm_model:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
            else:
                input_ids, token_type_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            val_probs += torch.softmax(outputs.logits, -1).tolist()

    acc = accuracy_score(val_labels, np.array(val_probs).argmax(-1))
    f1 = f1_score(val_labels, np.array(val_probs).argmax(-1))
    
    best_probs = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            if 'roberta' in args.lm_model:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
            else:
                input_ids, token_type_ids, attention_mask = batch
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
            best_probs += torch.softmax(outputs.logits, -1).tolist()
    
    current_best_probs = VectorScaling(verbose=False)(
                                valid_preacts=np.array(val_probs),
                                valid_labels=F.one_hot(torch.tensor(val_labels)).numpy(),
                                posterior_supplied=True)(best_probs)
    
    if probs_not_usable(current_best_probs, val_labels):
        current_best_probs = NoBiasVectorScaling(verbose=False)(
                                    valid_preacts=np.array(val_probs),
                                    valid_labels=F.one_hot(torch.tensor(val_labels)).numpy(),
                                    posterior_supplied=True)(best_probs)

    # if probs_not_usable(current_best_probs, val_labels):
    #     current_best_probs = IsotonicRegression(verbose=False)(
    #                                 valid_preacts=np.array(val_probs),
    #                                 valid_labels=np.array(val_labels)
    #                                 )(np.array(best_probs))

    indices = np.array(current_best_probs).max(-1) >= conf_threshold
    pseudolabels = np.array(current_best_probs).argmax(-1)[indices]
    if probs_not_usable(current_best_probs, val_labels):
        neg_probs = np.array(best_probs)[:, 0]
        min_neg_prob = np.sort(neg_probs)[::-1][int((np.array(val_labels)==0).mean()*len(neg_probs))]
        neg_probs = neg_probs * (neg_probs > min(0.5, min_neg_prob))
        neg_poses = neg_probs.nonzero()[0]
        pos_probs = np.array(best_probs)[:, 1]
        min_pos_prob = np.sort(pos_probs)[::-1][int((np.array(val_labels)==1).mean()*len(pos_probs))]
        pos_probs = pos_probs * (pos_probs > min(0.5, min_pos_prob))
        pos_poses = pos_probs.nonzero()[0]
        val_ratio = (np.array(val_labels) == 0).sum() / (np.array(val_labels) == 1).sum()
        indices_0 = np.random.choice(neg_poses, int((1-conf_threshold)*min(len(neg_poses), int(len(pos_poses)*val_ratio))), \
                                     replace=False, p=neg_probs[neg_poses]/neg_probs[neg_poses].sum())
        indices_1 = np.random.choice(pos_poses, int((1-conf_threshold)*min(len(pos_poses), int(len(neg_poses)/val_ratio))), \
                                     replace=False, p=pos_probs[pos_poses]/pos_probs[pos_poses].sum())
        indices = [True if (i in indices_0 or i in indices_1) else False for i in range(len(current_best_probs))]
        pseudolabels = np.array([0 if i in indices_0 else 1 for i in range(len(indices)) if (i in indices_0 or i in indices_1)])
        print('***** Scaling fails, roll back to confidence sampling *****')

    test_texts = np.array(test_texts)[indices]

    print('***** Input {} examples. Output {} samples, {} positive and {} negative. *****'.format(
        len(current_best_probs), len(pseudolabels), (pseudolabels == 1).sum(), (pseudolabels == 0).sum()
    ))
    return test_texts, pseudolabels


if __name__=="__main__":
    source_types = ['liar']
    source_paths = ['./data/LIAR']
    target_types = ['constraint']
    target_paths = ['./data/Constraint']
    get_psuedolabels_tfidf(source_types, source_paths, target_types, target_paths, conf_threshold=0.65)