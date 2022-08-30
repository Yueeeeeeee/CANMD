import pandas as pd
import numpy as np

import re
import html
import emoji
import unicodedata
import unidecode
import preprocessor as p
from string import punctuation

from config import args
from dataset import get_dataset

import torch
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


# def preprocess(args, data):
#     control_char_regex = re.compile(r'[\r\n\t]+')
#     transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
#     giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
#         '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

#     data["label"] = data["label"].map({"real": 1, "fake": 0})
#     data = data.drop(["id"], axis=1).values
#     for i in range(data.shape[0]):
#         text = data[i, 0]
#         text = html.unescape(text)
#         text = text.translate(transl_table)
#         text = text.replace('…', '...')
#         text = re.sub(control_char_regex, ' ', text)
#         text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
#         text = ' '.join(text.split())
#         text = text.strip()
        
#         text = re.sub(giant_url_regex, 'url', text)
#         text = text.replace('httpurl', 'url')
#         text = emoji.demojize(text)

#         text = unidecode.unidecode(text)
#         text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
#         data[i, 0] = text.strip()

#     return data[:, 0], data[:, 1].astype(int)


def preprocess(args, data):
    control_char_regex = re.compile(r'[\r\n\t]+')
    transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
    special_symbols = ['@', '#']

    options = []
    if args.tokenize_url:
        options.append(p.OPT.URL)
    if args.tokenize_emoji:
        options.append(p.OPT.EMOJI)
    if args.tokenize_smiley:
        options.append(p.OPT.SMILEY)
    if args.tokenize_hashtag:
        options.append(p.OPT.HASHTAG)
    if args.tokenize_mention:
        options.append(p.OPT.MENTION)
    if args.tokenize_number:
        options.append(p.OPT.NUMBER)
    if args.tokenize_reserved:
        options.append(p.OPT.RESERVED)
    if args.remove_escape_char:
        options.append(p.OPT.ESCAPE_CHAR)

    p.set_options(*options)

    for i in range(len(data)):
        text = data[i]
        text = html.unescape(text)
        text = re.sub(control_char_regex, ' ', text)
        text = text.translate(transl_table)
        text = text.replace('…', '...')
        text = unidecode.unidecode(text)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')

        if args.translate_emoji:
            text = emoji.demojize(text)
        if args.separate_special_symbol:
            for symbol in special_symbols:
                text = text.replace(symbol, ' '+symbol+' ')
        
        text = p.tokenize(text)
        if args.remove_url:
            text.replace('$URL$', '')
        if args.remove_emoji:
            text.replace('$EMOJI$', '')
        if args.remove_smiley:
            text.replace('$SMILEY$', '')
        if args.remove_hashtag:
            text.replace('$HASHTAG$', '')
        if args.remove_mention:
            text.replace('$MENTION$', '')
        if args.remove_number:
            text.replace('$NUMBER$', '')
        if args.remove_reserved:
            text.replace('$RESERVED$', '')

        data[i] = re.sub(' +', ' ', text.strip())
    
    return data


def tokenize(args, data, tokenizer=None):
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    if tokenizer is None:
        if 'roberta' in args.lm_model:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            tokenizer = BertTokenizer.from_pretrained(args.lm_model, do_lower_case=args.do_lower_case)
    
    added_tokens = []
    if args.tokenize_url:
        added_tokens.append('$URL$')
    if args.tokenize_emoji:
        added_tokens.append('$EMOJI$')
    if args.tokenize_smiley:
        added_tokens.append('$SMILEY$')
    if args.tokenize_hashtag:
        added_tokens.append('$HASHTAG$')
    if args.tokenize_mention:
        added_tokens.append('$MENTION$')
    if args.tokenize_number:
        added_tokens.append('$NUMBER$')
    if args.tokenize_reserved:
        added_tokens.append('$RESERVED$')
    
    if len(added_tokens) > 0:
        if 'roberta' not in args.lm_model and args.do_lower_case:
            added_tokens = [x.lower() for x in added_tokens]
        added_tokens = [x for x in added_tokens if x not in tokenizer.get_vocab().keys()]
        tokenizer.add_tokens(added_tokens)
    
    for input_text in data:
        encoded_input = tokenizer.encode_plus(
                            input_text,                      # Sentence to encode.
                            add_special_tokens=True,         # Add '[CLS]' and '[SEP]'
                            max_length=args.max_seq_length,  # Maximum length
                            padding='max_length',            # Pad to maximum length
                            truncation=True,                 # Truncate when necessary
                            return_attention_mask = True,    # Construct attention masks
                            return_tensors = 'pt'            # Return pytorch tensors.
                            )
        all_input_ids.append(encoded_input['input_ids'])
        if 'roberta' not in args.lm_model:
            all_token_type_ids.append(encoded_input['token_type_ids'])
        all_attention_mask.append(encoded_input['attention_mask'])
    
    all_input_ids = torch.vstack(all_input_ids)
    if 'roberta' not in args.lm_model:
        all_token_type_ids = torch.vstack(all_token_type_ids)
    all_attention_mask = torch.vstack(all_attention_mask)

    return all_input_ids, all_token_type_ids, all_attention_mask


def get_loader(args, mode, tokenizer=None):
    assert mode in ('source_train', 'source_val', 'source_test', 'target_train', 'target_val', 'target_test')
    if mode == 'source_train':
        data, labels = get_dataset(args, 'train', args.source_data_type, args.source_data_path).load_dataset()
    if mode == 'source_val':
        data, labels = get_dataset(args, 'val', args.source_data_type, args.source_data_path).load_dataset()
    if mode == 'source_test':
        data, labels = get_dataset(args, 'test', args.source_data_type, args.source_data_path).load_dataset()
    if mode == 'target_train':
        data, labels = get_dataset(args, 'train', args.target_data_type, args.target_data_path).load_dataset()
    elif mode == 'target_val':
        data, labels = get_dataset(args, 'val', args.target_data_type, args.target_data_path).load_dataset()
    elif mode == 'target_test':
        data, labels = get_dataset(args, 'test', args.target_data_type, args.target_data_path).load_dataset()
    
    inputs = preprocess(args, data)
    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    
    if 'roberta' not in args.lm_model:
        dataset = TensorDataset(
                    all_input_ids,
                    all_token_type_ids,
                    all_attention_mask,
                    torch.tensor(labels)
                    )
    else:
        dataset = TensorDataset(
                    all_input_ids,
                    all_attention_mask,
                    torch.tensor(labels)
                    )
    
    if mode in ('source_train', 'target_train'):
        sampler = RandomSampler(dataset)
        batch_size = args.train_batchsize
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batchsize
    
    loader = DataLoader(
                dataset,  
                sampler=sampler,
                batch_size=batch_size
                )

    return loader


# loader = get_loader(args, 'source_train')
# for i, batch in enumerate(loader):
#     print(batch[0])