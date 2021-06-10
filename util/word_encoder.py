import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True, return_dict=True)
        #outputs = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        word_embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        return word_embeddings
    
    def tokenize(self, raw_tokens, tags):
        raw_tokens = [token.lower() for token in raw_tokens]
        indexed_tokens_list = []
        tag_list = []
        mask_list = []
        text_mask_list = []
        
        curr_split = ['[CLS]']
        tag_split = []
        mask_split = np.zeros((self.max_length), dtype=np.int32)
        text_mask_split = np.zeros((self.max_length), dtype=np.int32)
        
        for i, (word, tag) in enumerate(zip(raw_tokens, tags)):
            tokens = self.tokenizer.tokenize(word)
            
            if len(curr_split) + len(tokens) >= self.max_length:
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(curr_split + ['[SEP]'])
                while len(indexed_tokens) < self.max_length:
                    indexed_tokens.append(0)
                mask_split[:len(indexed_tokens)] = 1
                
                indexed_tokens_list.append(indexed_tokens)
                tag_list.append(tag_split)
                mask_list.append(mask_split)
                text_mask_list.append(text_mask_split)
                
                curr_split = ['[CLS]']
                tag_split = []
                mask_split = np.zeros((self.max_length), dtype=np.int32)
                text_mask_split = np.zeros((self.max_length), dtype=np.int32)
            
            text_mask_split[len(curr_split)] = 1
            curr_split.extend(tokens)
            tag_split.append(tag)
                
        
        if tag_split:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(curr_split + ['[SEP]'])
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            mask_split[:len(indexed_tokens)] = 1

            indexed_tokens_list.append(indexed_tokens)
            tag_list.append(tag_split)
            mask_list.append(mask_split)
            text_mask_list.append(text_mask_split)
        
        # print ("tokens   :", indexed_tokens_list[0])
        # print ("masks    :", mask_list[0])
        # print ("text_mask:", text_mask_list[0])
        # print ("tag_list :", tag_list[0])
        # print ()
        
        return indexed_tokens_list, mask_list, text_mask_list, tag_list
