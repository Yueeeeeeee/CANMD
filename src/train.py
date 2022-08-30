import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

from dataset import get_dataset
from dataloader import preprocess, tokenize, get_loader
from config import *
from tqdm import tqdm
import json
import os


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation(args, model, eval_dataloader):
    model.eval()
    eval_preds = []
    eval_labels = []
    eval_losses = []
    
    tqdm_dataloader = tqdm(eval_dataloader)
    for _, batch in enumerate(tqdm_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        if 'roberta' in args.lm_model:
            input_ids, attention_mask, labels = batch
            with torch.no_grad():        
                outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                        )
        else:
            input_ids, token_type_ids, attention_mask, labels = batch
            with torch.no_grad():        
                outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels
                        )

        loss = outputs[0]
        logits = outputs[1]
        eval_preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
        eval_labels += labels.cpu().numpy().tolist()
        eval_losses.append(loss.item())

        tqdm_dataloader.set_description('Eval bacc: {:.4f}, acc: {:.4f}, f1: {:.4f}, loss: {:.4f}'.format(
            balanced_accuracy_score(eval_labels, eval_preds),
            np.mean(np.array(eval_labels)==np.array(eval_preds)), 
            f1_score(eval_labels, eval_preds),
            np.mean(eval_losses)
        ))

    final_bacc = balanced_accuracy_score(eval_labels, eval_preds)
    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    final_f1 = f1_score(eval_labels, eval_preds)
    final_precision = precision_score(eval_labels, eval_preds)
    final_recall = recall_score(eval_labels, eval_preds)
    
    return final_bacc, final_acc, final_f1, final_precision, final_recall


def train(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.output_dir:
        args.output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
    export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model,
        do_lower_case=args.do_lower_case
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.lm_model,
        num_labels=2, 
        output_attentions=False, 
        output_hidden_states=False
        )

    train_dataloader = get_loader(args, mode='source_train', tokenizer=tokenizer)
    val_dataloader = get_loader(args, mode='source_val', tokenizer=tokenizer)
    test_dataloader = get_loader(args, mode='source_test', tokenizer=tokenizer)
    t_total = len(train_dataloader) * args.num_train_epochs

    model.resize_token_embeddings(len(tokenizer))
    try:
        model = model.from_pretrained(
            export_root,
            num_labels=2, 
            output_attentions=False, 
            output_hidden_states=False
            ).to(args.device)
        print('***** Previous model found, continue training *****')
    except:
        print('***** Previous model not found, train from scratch *****')
    model.to(args.device)

    if 'roberta' in args.lm_model:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                            num_warmup_steps=args.warmup_proportion*t_total,
                            num_training_steps=t_total)
    
    global_step = 0
    print('***** Running training *****')
    print('Batch size = %d', args.train_batchsize)
    print('Num steps = %d', t_total)
    best_bacc, best_acc, best_f1, _, _ = evaluation(args, model, val_dataloader)
    for epoch in range(1, args.num_train_epochs+1):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc='Epoch {}'.format(epoch))):
            batch = tuple(t.to(args.device) for t in batch)
            if 'roberta' in args.lm_model:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            else:
                input_ids, token_type_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if 'roberta' not in args.lm_model:
                scheduler.step()
            model.zero_grad()
            global_step += 1

        eval_bacc, eval_acc, eval_f1, _, _ = evaluation(args, model, val_dataloader)
        if eval_bacc + eval_acc + eval_f1 >= best_bacc + best_acc + best_f1:
            best_bacc = eval_bacc
            best_acc = eval_acc
            best_f1 = eval_f1
            print('***** Saving best model *****')
            model.save_pretrained(export_root)
            tokenizer.save_pretrained(export_root)
            output_args_file = os.path.join(export_root, 'training_args.bin')
            torch.save(args, output_args_file)
    
    print('***** Running evaluation *****')
    model = AutoModelForSequenceClassification.from_pretrained(
        export_root,
        num_labels=2, 
        output_attentions=False, 
        output_hidden_states=False
        ).to(args.device)
    test_bacc, test_acc, test_f1, test_precision, test_recall = evaluation(args, model, test_dataloader)
    with open(os.path.join(export_root, 'test_metrics.json'), 'w') as f:
        json.dump({
            'bacc': test_bacc,
            'acc': test_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
            }, f)


if __name__ == '__main__': 
    args.num_train_epochs = 5  # number of epoch can be chosen from 2 to 5
    train(args)
