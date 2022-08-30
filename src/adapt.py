import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from model_bert import ContrastiveBertForSequenceClassification as CDABERT
from model_roberta import ContrastiveRobertaForSequenceClassification as CDARoberta

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

from dataset import get_dataset
from dataloader import preprocess, tokenize, get_loader
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from config import *
from utils import *
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


def adapt(args):
    def sample_source_batch(args, target_labels, source_dataset, source_label_dict, source_pointer):
        output_idx = []
        for label in target_labels.tolist():
            next_idx = source_pointer[label] % len(source_label_dict[label])
            output_idx.append(source_label_dict[label][next_idx])
            source_pointer[label] += 1

        all_input_ids, all_token_type_ids, all_attention_mask, labels  = [], [], [], []
        for idx in output_idx:
            all_input_ids.append(source_dataset[idx][0].unsqueeze(0))
            if 'roberta' not in args.lm_model:
                all_token_type_ids.append(source_dataset[idx][1].unsqueeze(0))
            all_attention_mask.append(source_dataset[idx][-2].unsqueeze(0))
            labels.append(source_dataset[idx][-1].unsqueeze(0))
        
        all_input_ids = torch.vstack(all_input_ids)
        if 'roberta' not in args.lm_model:
            all_token_type_ids = torch.vstack(all_token_type_ids)
        all_attention_mask = torch.vstack(all_attention_mask)
        labels = torch.vstack(labels).squeeze()

        if 'roberta' not in args.lm_model:
            return all_input_ids, all_token_type_ids, all_attention_mask, labels
        else:
            return all_input_ids, all_attention_mask, labels
    
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
    if 'roberta' in args.lm_model:
        model = CDARoberta.from_pretrained(
            args.lm_model,
            num_labels=2, 
            output_attentions=False, 
            output_hidden_states=False
            )
    else:
        model = CDABERT.from_pretrained(
            args.lm_model,
            num_labels=2, 
            output_attentions=False, 
            output_hidden_states=False
            )

    source_pointer = [0] * 2
    source_label_dict = {0: [], 1: []}
    data, labels = get_dataset(args, 'train', args.source_data_type, args.source_data_path).load_dataset()
    for idx, label in enumerate(labels):
        source_label_dict[label].append(idx)
    for key in source_label_dict.keys():
        random.shuffle(source_label_dict[key])
    
    inputs = preprocess(args, data)
    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    if 'roberta' in args.lm_model:
        source_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, torch.tensor(labels))
    else:
        source_dataset = torch.utils.data.TensorDataset(all_input_ids, all_token_type_ids, \
                            all_attention_mask, torch.tensor(labels))
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset, sampler=RandomSampler(source_dataset), batch_size=args.train_batchsize)
    
    val_dataloader = get_loader(args, mode='target_val', tokenizer=tokenizer)
    test_dataloader = get_loader(args, mode='target_test', tokenizer=tokenizer)
    t_total = len(source_dataloader) * args.num_train_epochs  # notice t_total is not be accurate
    
    if not args.load_model_path:
        args.load_model_path = os.path.join(EXPERIMENT_ROOT_FOLDER, args.source_data_type)
    else:
        args.load_model_path = os.path.join(EXPERIMENT_ROOT_FOLDER, args.load_model_path)
    if os.path.isfile(os.path.join(args.load_model_path, 'pytorch_model.bin')):
        try:
            model = model.from_pretrained(
                        os.path.join(EXPERIMENT_ROOT_FOLDER, args.source_data_type),
                        num_labels=2, 
                        output_attentions=False, 
                        output_hidden_states=False
                        )
            model.resize_token_embeddings(len(tokenizer))
        except:
            model.resize_token_embeddings(len(tokenizer))
            model = model.from_pretrained(
                        os.path.join(EXPERIMENT_ROOT_FOLDER, args.source_data_type),
                        num_labels=2, 
                        output_attentions=False, 
                        output_hidden_states=False
                        )
        print('Source trained model is loaded...')
    else:
        model.resize_token_embeddings(len(tokenizer))
        print('Source trained model is not found...')
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
    print('***** Running adaptation *****')
    print('Batch size = {}'.format(args.train_batchsize))
    print('Num steps = {}'.format(t_total))
    best_bacc, best_acc, best_f1, _, _ = evaluation(args, model, val_dataloader)
    for epoch in range(1, args.num_train_epochs+1):
        filtered_data, pseudolabels = get_corrected_psuedolabels_model(
                                        args, tokenizer, model, 
                                        [args.target_data_type], [args.target_data_path], 
                                        conf_threshold=args.conf_threshold)
        all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, filtered_data, tokenizer)
        if 'roberta' in args.lm_model:
            target_dataset = torch.utils.data.TensorDataset(
                    all_input_ids,
                    all_attention_mask,
                    torch.tensor(pseudolabels))
        else:
            target_dataset = torch.utils.data.TensorDataset(
                    all_input_ids,
                    all_token_type_ids,
                    all_attention_mask,
                    torch.tensor(pseudolabels))
        
        target_dataloader = DataLoader(
                target_dataset,  
                sampler=RandomSampler(target_dataset),
                batch_size=args.train_batchsize)

        model.train()
        for step, batch in enumerate(tqdm(target_dataloader, desc='Epoch {}'.format(epoch))):
            source_batch = sample_source_batch(args, batch[-1], source_dataset, source_label_dict, source_pointer)
            source_batch = tuple(t.to(args.device) for t in source_batch)
            target_batch = tuple(t.to(args.device) for t in batch)
            
            if 'roberta' in args.lm_model:
                source_input_ids, source_attention_mask, source_labels = source_batch
                target_input_ids, target_attention_mask, target_labels = target_batch
                outputs = model.forward_ours(
                    input_ids=torch.vstack((source_input_ids, target_input_ids)),
                    attention_mask=torch.vstack((source_attention_mask, target_attention_mask)),
                    labels=torch.vstack((source_labels, target_labels)),
                    alpha=args.alpha)
            else:
                source_input_ids, source_token_type_ids, source_attention_mask, source_labels = source_batch
                target_input_ids, target_token_type_ids, target_attention_mask, target_labels = target_batch
                outputs = model.forward_ours(
                    input_ids=torch.vstack((source_input_ids, target_input_ids)),
                    attention_mask=torch.vstack((source_attention_mask, target_attention_mask)),
                    token_type_ids=torch.vstack((source_token_type_ids, target_token_type_ids)),
                    labels=torch.vstack((source_labels, target_labels)),
                    alpha=args.alpha)

            loss = outputs[0]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if 'roberta' not in args.lm_model:
                scheduler.step()
            model.zero_grad()
            global_step += 1

        eval_bacc, eval_acc, eval_f1, _, _ = evaluation(args, model, val_dataloader)
        if eval_bacc + eval_acc + eval_f1 >= best_bacc + best_acc + best_f1:  # use sum to validate model
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
    print(args)
    adapt(args)
