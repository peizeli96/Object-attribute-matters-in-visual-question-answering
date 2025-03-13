from __future__ import print_function
import os
from torch.nn import functional as F
import json
import _pickle as cPickle
import numpy as np
import pickle
import utils as utils
import gzip
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import random
COUNTING_ONLY = False
from transformers import BertTokenizer




class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['img_id'],
        'image': img,
        'question': question['question'],
        'answer': answer,
        'answer_token': question['answer']}
    return entry



def get_ques_ans_path(name):
    

    if name == 'train':
        question_path = '../data/train.json'
        answer_path = os.path.join('../data', 'train_target.pkl')
    elif name == 'valid':
        question_path = '../data/valid.json'
        answer_path = os.path.join('../data', 'valid_target.pkl')
    elif name == 'test':
        question_path = '../data/test.json'
        answer_path = os.path.join('../data', 'test_target.pkl')

    else:
        print('plz set name is one of [train, valid, test]!')
        assert 1==2

    return question_path, answer_path


def _load_qa_dataset(dataroot, name):
    # """Load entries
    # img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    # dataroot: root path of dataset
    # name: 'train', 'val', 'test-dev2015', test2015'
    # """
    question_path, answer_path = get_ques_ans_path(name)
    questions = sorted(json.load(open(question_path)), key=lambda x: x['question_id'])

    # train, val
    answers = cPickle.load(open(answer_path, 'rb'))
    # import pdb;pdb.set_trace()
    answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]
    utils.assert_eq(len(questions), len(answers))

    entries = []
    for question, answer in zip(questions, answers):
        # import pdb;pdb.set_trace()
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['img_id'], answer['image_id'])
        img_id = question['img_id']

        entries.append(_create_entry(img_id, question, answer))
    return entries





class VQAFeatureDataset(Dataset):
    def __init__(self, opt, name, dictionary):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'valid', 'test']


        ans2label_path = os.path.join('../data/', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join('../data/', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary

        print('loading image features and bounding boxes')
        # Load image features and bounding boxes
        
        with open('../data/_img_feature_%s.pickle' % name, 'rb') as f:
            self.pretrain_feature = pickle.load(f)

        print('loading image features and bounding boxes done!')

        self.entries = _load_qa_dataset("../data", name)

        self.kb_features = json.load(open('../data/%s_kb.json' % name)) 
        self.kb_token = {}
        

        self.konw = json.load(open('../data/kb_%s_krvqa_dep.json'% name))
        self.kb = {}
        
        self.attribute = json.load(open("../data/%s_attribute.json" % name))
        self.attribute_feature = {}

        self.lxmert_tokenize(18)
        self.blip_tokenize(1)
        self.kb_tokenize(10)
        self.attribute_tokenize(40)


        self.tensorize(name)  




    def lxmert_tokenize(self, max_length=18):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """

        for entry in self.entries:
            lower_question_text = entry['question'] 
            tokens = self.dictionary.tokenize(lower_question_text, False)

            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            else:
                tokens = tokens[:max_length] 
            
            entry['sent'] = tokens


    def kb_tokenize(self, max_length=6):

        for qid in self.konw:
            lxmert_word = self.konw[qid]
            # import pdb; pdb.set_trace()
            tokens = self.dictionary.tokenize(lxmert_word, False)

            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            else:
                tokens = tokens[:max_length] 
            self.kb[qid] = tokens


    def blip_tokenize(self, max_length=14):

        for qid in self.blip_features:
            lxmert_word = self.blip_features[qid]
            tokens = self.dictionary.tokenize(lxmert_word, False)

            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            else:
                tokens = tokens[:max_length] 
                
            self.blip_token[qid] = tokens
   
            
            
            
            


    def attribute_tokenize(self, max_length=18):

        for qid in self.attribute:
            lxmert_word_list = self.attribute[qid]   # 18 * a
            # import pdb; pdb.set_trace()
            tokens = self.dictionary.tokenize(" ".join(lxmert_word_list), False)

            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            else:
                tokens = tokens[:max_length] 

            self.attribute_feature[qid] = tokens


    def tensorize(self, name):

        print('data',len(self.entries))
        
        for entry in self.entries:
            answer = entry['answer']

            ans_token = torch.from_numpy(np.array(answer['labels']))
            entry['answer_token'] = ans_token            

            
            sent = torch.from_numpy(np.array(entry['sent']))
            entry['sent'] = sent
            

            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    # print(labels)                    
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['sent']
        question_id = entry['question_id']
        answer = entry['answer']  
        img_id = str(entry['image_id'])

        features = torch.from_numpy(np.array(self.pretrain_feature[img_id]['feats'], dtype=float)).float()
        spatials = torch.from_numpy(np.array(self.pretrain_feature[img_id]['sp_feats'], dtype=float)).float()

        blip_token = torch.from_numpy(np.array(self.blip_token[question_id]))
        kb_token = torch.from_numpy(np.array(self.kb[question_id]))
        attribute_token = torch.from_numpy(np.array(self.attribute_feature[img_id]))
                                    

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)


            return features, spatials, question, target, question_id, kb_token, attribute_token, blip_token
        else:

            return features, spatials, question, question_id, kb_token, attribute_token, blip_token
        
    def __len__(self):
        return len(self.entries)


