import json
import torch

from Pretraining.utils import *
from Pretraining.model import RelationPT
from Pretraining.model_rob import RelationPT_rob

from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)
from transformers import (RobertaConfig, RobertaModel,AutoTokenizer, RobertaPreTrainedModel)

import argparse

import numpy as np
from tqdm import tqdm
# from fuzzywuzzy import fuzz
import os
import pickle

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


## Todo changed
#tokenizer = BertTokenizer.from_pretrained('/data/csl/resources/Bert/bert-base-cased', do_lower_case = False)
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
import random

def decision(probability):
    return random.random() < probability


# def load_model():
#
#     """
#     Load the model from checkpoint
#     """
#
#     #save_dir = "PaulD/checkpoint-14399"
#     save_dir = "PaulD/IOA_261022-11999"
#     config_class, model_class = (BertConfig, RelationPT)
#     print("load ckpt from {}".format(save_dir))
#     config = config_class.from_pretrained(save_dir)  # , num_labels = len(label_list))
#     model = model_class.from_pretrained(save_dir, config=config)
#     #path = './processed/vocab.json'
#     #model.config.vocab = load_vocab(path)
#
#     n_gpu = torch.cuda.device_count()
#     if torch.cuda.is_available():  #
#         model.cuda()
#         #if n_gpu > 1:
#         #    model = torch.nn.DataParallel(model)
#     return model


def get_vocab(args, vocab):
    kb = json.load(open(os.path.join(args.input_dir, args.kb)))#'esa_kb.json')))
    #kb = json.load(open(os.path.join(args.input_dir, 'kb.json')))
    entities = kb['entities']
    for eid in entities:
        relations = entities[eid]['relations']
        for relation in relations:
            if relation.get('relation') == None:
                r= relation['predicate']
            else:
                r = relation['relation']
            if relation['direction'] == 'backward':
                r = '[inverse] ' + r
            if not r in vocab['relation2id']:
                vocab['relation2id'][r] = len(vocab['relation2id'])
    vocab['id2relation'] = [relation for relation, id in vocab['relation2id'].items()]

    for eid in entities:

        ## Todo included check, some entities don't have name in current version, alternative to take
        if entities[eid].get('name') != None:
            entity = entities[eid]['name']
            # unique_ent = [kb['entities'][entity]['name'] for entity in kb['entities'].keys()]
            if not entity in vocab['entity2id']:
                vocab['entity2id'][entity] = len(vocab['entity2id'])
        else:
            continue
    vocab['id2entity'] = [entity for entity, iid in vocab['entity2id'].items()]

    unique_atts = set([attribute['key'] for entity in kb['entities'].keys() for attribute in kb['entities'][entity]['attributes']])
    for attribute in unique_atts:
        if not attribute in vocab['attribute2id']:
            vocab['attribute2id'][attribute] = len(vocab['attribute2id'])
    vocab['id2attribute'] = [entity for entity, iid in vocab['attribute2id'].items()]

    concepts = kb['concepts']
    for cid in concepts:
        concept = concepts[cid]['name']
        if not concept in vocab['concept2id']:
            vocab['concept2id'][concept] = len(vocab['concept2id'])
    vocab['id2concept'] = [concept for concept, id in vocab['concept2id'].items()]


    ## Todo commented out for now because not train set
    # train = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'train.json'))][0]
    # for item in train:
    #     program = item['program']
    #     for f in program:
    #         function = f['function']
    #         if not function in vocab['function2id']:
    #             vocab['function2id'][function] = len(vocab['function2id'])

    vocab['id2function'] = [function for function, id in vocab['function2id'].items()]
    vocab['id2operator'] = [function for function, id in vocab['operator2id'].items()]

def get_relation_dataset(args, vocab):
    ## Todo changed

    # train = json.load(open(os.path.join(args.input_dir, 'train.json')))
    # dev = json.load(open(os.path.join(args.input_dir, 'val.json')))
    train = [json.loads(line.strip()) for line in open(os.path.join(args.train_file_path))][0]
    dev = [json.loads(line.strip()) for line in open(os.path.join(args.valid_file_path))][0]

    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            #print(item)
            text = item['question']
            program = item['program']
            data = []
            relations = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'Relate':
                    inputs = f['inputs']
                    r = inputs[0]
                    if inputs[1] == 'backward':
                        r = '[inverse] ' + r
                    if not r in vocab['relation2id']:
                        continue
                    r = vocab['relation2id'][r]
                    relations.append([idx + 1, r])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(relations) == 0:
                relations.append([0, vocab['relation2id']['<PAD>']])
            dataset.append({'question': text, 'program': data, 'relations': relations})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['relations']:
        #             print(pos, vocab['id2relation'][r])


        with open(os.path.join(args.output_dir, 'relation', '%s.json'%(name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def get_concept_dataset(args, vocab):

    ## Todo changed
    # train = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'train.json'))]
    # dev = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'val.json'))]

    train = [json.loads(line.strip()) for line in open(args.train_file_path)][0]
    dev = [json.loads(line.strip()) for line in open(args.valid_file_path)][0]
    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            text = item['question']
            program = item['program']
            data = []
            concepts = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'FilterConcept':
                #if function == 'FilterConcept'
                    inputs = f['inputs']
                    c = inputs[0]
                    if not c in vocab['concept2id']:
                        continue
                    c = vocab['concept2id'][c]
                    concepts.append([idx + 1, c])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(concepts) == 0:
                concepts.append([0, vocab['concept2id']['<PAD>']])
            dataset.append({'question': text, 'program': data, 'concepts': concepts})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['concepts']:
        #             print(pos, vocab['id2concept'][r])


        with open(os.path.join(args.output_dir, 'concept', '%s.json'%(name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def get_entity_dataset(args, vocab):

    ## Todo changed
    # train = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'train.json'))]
    # dev = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'val.json'))]
    train = [json.loads(line.strip()) for line in open(args.train_file_path)][0]
    dev = [json.loads(line.strip()) for line in open(args.valid_file_path)][0]

    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            text = item['question']
            program = item['program']
            data = []
            concepts = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'Find':
                #if function == 'FilterConcept'
                    inputs = f['inputs']
                    c = inputs[0]
                    if not c in vocab['entity2id']:
                        continue
                    c = vocab['entity2id'][c]
                    concepts.append([idx + 1, c])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(concepts) == 0:
                concepts.append([0, vocab['entity2id']['<PAD>']])
            dataset.append({'question': text, 'program': data, 'entities': concepts})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['concepts']:
        #             print(pos, vocab['id2concept'][r])


        with open(os.path.join(args.output_dir, 'entity', '%s.json'%(name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def get_operator_dataset(args, vocab):
    train = [json.loads(line.strip()) for line in open(args.train_file_path)][0]
    dev = [json.loads(line.strip()) for line in open(args.valid_file_path)][0]

    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            text = item['question']
            program = item['program']
            data = []
            relations = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'FilterYear' or function == 'FilterData' or function == 'FilterNum':
                    inputs = f['inputs']
                    r = inputs[2]
                    if not r in vocab['operator2id']:
                        continue
                    r = vocab['operator2id'][r]
                    relations.append([idx + 1, r])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(relations) == 0:
                if decision(0.05):
                    relations.append([0, vocab['operator2id']['<PAD>']])
                else:
                    continue
            dataset.append({'question': text, 'program': data, 'operations': relations})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['relations']:
        #             print(pos, vocab['id2relation'][r])
        with open(os.path.join(args.output_dir, 'operator', '%s.json' % (name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def get_attribute_dataset(args, vocab):

    train = [json.loads(line.strip()) for line in open(args.train_file_path)][0]
    dev = [json.loads(line.strip()) for line in open(args.valid_file_path)][0]

    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            text = item['question']
            program = item['program']
            data = []
            concepts = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'QueryAttr' or function == 'FilterYear' or function == 'FilterData' or function == 'FilterNum':
                    # if function == 'FilterConcept'
                    inputs = f['inputs']
                    c = inputs[0]
                    if not c in vocab['attribute2id']:
                        continue
                    c = vocab['attribute2id'][c]
                    concepts.append([idx + 1, c])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(concepts) == 0:
                concepts.append([0, vocab['attribute2id']['<PAD>']])
            dataset.append({'question': text, 'program': data, 'attributes': concepts})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['concepts']:
        #             print(pos, vocab['id2concept'][r])

        with open(os.path.join(args.output_dir, 'attribute', '%s.json' % (name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def encode_kb_entity(args, vocab, pred_type, tokenizer):

    encoded_inputs = tokenizer(vocab[f'id2{pred_type}'], padding=True, return_token_type_ids = True)
    print(encoded_inputs.keys())
    print(len(encoded_inputs['input_ids'][0]))
    print(len(encoded_inputs['token_type_ids'][0]))
    print(len(encoded_inputs['attention_mask'][0]))
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    return input_ids_list, token_type_ids_list, attention_mask_list

def encode_relation_dataset(args, vocab, dataset, tokenizer):
    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        relations = item['relations']
        for relation in relations:
            tmp.append({'question': question, 'program': program, 'relation': relation})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding = True, return_token_type_ids = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [{'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['relation']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    # verbose = False
    # if verbose:
    #     for idx in range(10):
    #         question = tokenizer.decode(input_ids_list[idx])
    #         functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
    #         relation_pos = relation_pos_list[idx][0]
    #         relation_id = vocab['id2relation'][relation_id_list[idx][0]]
    #         print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)

    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list


def encode_operator_dataset(args, vocab, dataset, tokenizer):
    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        relations = item['operations']
        for relation in relations:
            tmp.append({'question': question, 'program': program, 'operation': relation})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding=True, return_token_type_ids = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [
            {'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['operation']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    # verbose = False
    # if verbose:
    #     for idx in range(10):
    #         question = tokenizer.decode(input_ids_list[idx])
    #         functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
    #         relation_pos = relation_pos_list[idx][0]
    #         relation_id = vocab['id2relation'][relation_id_list[idx][0]]
    #         print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)

    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list


def encode_concept_dataset(args, vocab, dataset,tokenizer):
    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        concepts = item['concepts']
        for concept in concepts:
            tmp.append({'question': question, 'program': program, 'concept': concept})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding = True, return_token_type_ids = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [{'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['concept']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    verbose = False
    if verbose:
        for idx in range(10):
            question = tokenizer.decode(input_ids_list[idx])
            functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
            relation_pos = relation_pos_list[idx][0]
            relation_id = vocab['id2concept'][relation_id_list[idx][0]]
            print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)

    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list


def encode_entity_dataset(args, vocab, dataset, tokenizer):
    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        concepts = item['entities']
        for concept in concepts:
            tmp.append({'question': question, 'program': program, 'entity': concept})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding=True, return_token_type_ids = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [
            {'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['entity']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    verbose = False
    if verbose:
        for idx in range(10):
            question = tokenizer.decode(input_ids_list[idx])
            functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
            relation_pos = relation_pos_list[idx][0]
            relation_id = vocab['id2concept'][relation_id_list[idx][0]]
            print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)

    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list


def encode_attribute_dataset(args, vocab, dataset,tokenizer):

    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        concepts = item['attributes']
        for concept in concepts:
            tmp.append({'question': question, 'program': program, 'attribute': concept})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding=True, return_token_type_ids = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [
            {'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['attribute']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    verbose = False
    if verbose:
        for idx in range(10):
            question = tokenizer.decode(input_ids_list[idx])
            functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
            relation_pos = relation_pos_list[idx][0]
            relation_id = vocab['id2attribute'][relation_id_list[idx][0]]
            print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)

    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list

def load_classes(path):
  with open(os.path.abspath(path), 'rb') as f:
      input_ids = pickle.load(f)
      token_type_ids = pickle.load(f)
      attention_mask = pickle.load(f)
      # input_ids = torch.LongTensor(input_ids[:512,:]).to(device)
      # token_type_ids = torch.LongTensor(token_type_ids[:512,:]).to(device)
      # attention_mask = torch.LongTensor(attention_mask[:512,:]).to(device)
      input_ids = torch.LongTensor(input_ids)#.to(device)
      token_type_ids = torch.LongTensor(token_type_ids)#.to(device)
      attention_mask = torch.LongTensor(attention_mask)#.to(device)
  argument_inputs = {
    'input_ids': input_ids,
    'token_type_ids': token_type_ids,
    'attention_mask': attention_mask
  }
  return argument_inputs

def embed_ents(model,args):
    batch_num = 128
    # argument_inputs = load_classes(input_dir + "esa/new/entity_3110.pt", )
    argument_inputs = load_classes(args.output_dir + "/entity/entity.pt")#,'cpu')
    data = TensorDataset(argument_inputs['input_ids'], argument_inputs['attention_mask'],
                         argument_inputs['token_type_ids'])
    data_sampler = SequentialSampler(data)
    dataloader = torch.utils.data.DataLoader(data, sampler=data_sampler, batch_size=batch_num)

    attribute_embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
           
            inputs = batch[0].to(device)
            masks = batch[1].to(device)
            tags = batch[2].to(device)

            if args.model_type == "bert":
                attribute_embeddings += model.bert(input_ids=inputs,
                                                   attention_mask=masks,
                                                   token_type_ids=tags)[1].cpu()

            elif args.model_type == "roberta":
                attribute_embeddings += model.roberta(input_ids=inputs,
                                                      attention_mask=masks,
                                                      token_type_ids=tags)[1].cpu()
    attribute_embeddings = torch.stack(attribute_embeddings)

    with open(os.path.join(args.output_dir, 'entity/entity_embeddings.pt'), 'wb') as f:
        pickle.dump(attribute_embeddings, f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required = True, type = str)
    parser.add_argument('--train_file_path', required = False, type = str, default= 'train.json')
    parser.add_argument('--valid_file_path', required=False, type=str, default= 'valid.json')
    parser.add_argument('--output_dir', required = True, type = str)
    parser.add_argument('--model_type', required = True, type= str, default= 'bert')
    parser.add_argument('--model_name', required = True, type=str, default= 'bert-base-case')
    parser.add_argument('--mode', required = True, type= str, default= '')
    parser.add_argument('--kb', required= True, type=str, default= 'esa_kb.json')

    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'relation')):
        os.makedirs(os.path.join(args.output_dir, 'relation'))
    if not os.path.isdir(os.path.join(args.output_dir, 'concept')):
        os.makedirs(os.path.join(args.output_dir, 'concept'))
    if not os.path.isdir(os.path.join(args.output_dir, 'entity')):
        os.makedirs(os.path.join(args.output_dir, 'entity'))
    if not os.path.isdir(os.path.join(args.output_dir, 'attribute')):
        os.makedirs(os.path.join(args.output_dir, 'attribute'))
    if not os.path.isdir(os.path.join(args.output_dir, 'operator')):
        os.makedirs(os.path.join(args.output_dir, 'operator'))

    vocab = {
        'relation2id': {
            '<PAD>': 0
        },
        'concept2id': {
            '<PAD>': 0
        },
        "function2id": {
        "<PAD>": 0,
        "<START>": 1,
        "<END>": 2,
        "FindAll": 3,
        "FilterStr": 4,
        "FilterConcept": 5,
        "And": 6,
        "What": 7,
        "Find": 8,
        "QueryAttrQualifier": 9,
        "Relate": 10,
        "QueryAttr": 11,
        "VerifyStr": 12,
        "FilterNum": 13,
        "SelectBetween": 14,
        "QueryRelationQualifier": 15,
        "QueryRelation": 16,
        "Count": 17,
        "VerifyNum": 18,
        "VerifyYear": 19,
        "FilterYear": 20,
        "SelectAmong": 21,
        "FilterDate": 22,
        "QueryAttrUnderCondition": 23,
        "QFilterYear": 24,
        "QFilterStr": 25,
        "Or": 26,
        "QFilterNum": 27,
        "QFilterDate": 28,
        "VerifyDate": 29
      },
      "operator2id": {'<PAD>': 0, '=': 1,
                      '<': 2, '!=': 3
                      , '>': 4
      },
        'entity2id': {
            '<PAD>': 0
         },
        'attribute2id': {
            '<PAD>': 0
                         }
    }
    get_vocab(args, vocab)

    #tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    # try:
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)#, do_lower_case=False)
    except:
        print("Could not load tokenizer from specified pretrained model")
        print("Loading Roberta tokenizer instead...")
        tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=False)

    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)

    #outputs = encode_relation(args, vocab)
    outputs = encode_kb_entity(args, vocab,'relation',tokenizer)
    with open(os.path.join(args.output_dir, 'relation', 'relation.pt'), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)

    #outputs = encode_entity(args, vocab)
    outputs = encode_kb_entity(args, vocab, 'entity',tokenizer)
    with open(os.path.join(args.output_dir, 'entity', 'entity.pt'), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)

    ## Loads model, needs filepath to already trained model, to create embeddings for entities 
    ## Requires GPU 
    if args.mode == 'inference':

        if args.model_type == 'bert':

            config_class, model_class = (BertConfig, RelationPT)

        elif args.model_type == 'roberta':
            
            config_class, model_class = (RobertaConfig, RelationPT_rob)

        print("load ckpt from {}".format(args.model_name))
        config = config_class.from_pretrained(args.model_name)  # , num_labels = len(label_list))
        model = model_class.from_pretrained(args.model_name, config=config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():  #
            model.cuda()
        ## Encodes entities into entity_embeddings.pt file and saves for inference
        embed_ents(model,args)

    #outputs = encode_concept(args, vocab)
    outputs = encode_kb_entity(args, vocab, 'concept', tokenizer)
    with open(os.path.join(args.output_dir, 'concept', 'concept.pt'), 'wb') as f:
        for o in outputs:
            print(tokenizer.decode(o[0])[0:2])
            print(o.shape)
            pickle.dump(o, f)

    outputs = encode_kb_entity(args, vocab, 'attribute', tokenizer)
    with open(os.path.join(args.output_dir, 'attribute', 'attribute.pt'), 'wb') as f:
        for o in outputs:
            print(tokenizer.decode(o[0][0:5]))
            print(o.shape)
            pickle.dump(o, f)
    if args.mode == 'train':
        get_relation_dataset(args, vocab)
        get_concept_dataset(args, vocab)
        get_entity_dataset(args, vocab)
        get_attribute_dataset(args, vocab)
        get_operator_dataset(args, vocab)
        # vocab = json.load(open(os.path.join(args.output_dir, 'vocab.json')))
        for name in ['train', 'dev']:
            dataset = []
            with open(os.path.join(args.output_dir, 'relation', '%s.json'%(name))) as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            outputs = encode_relation_dataset(args, vocab, dataset,tokenizer)
            assert len(outputs) == 6
            print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
            with open(os.path.join(args.output_dir, 'relation', '{}.pt'.format(name)), 'wb') as f:
                for o in outputs:
                    print(o.shape)
                    pickle.dump(o, f)

        for name in ['train', 'dev']:
            dataset = []
            with open(os.path.join(args.output_dir, 'concept', '%s.json'%(name))) as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            outputs = encode_concept_dataset(args, vocab, dataset,tokenizer)
            assert len(outputs) == 6
            print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
            with open(os.path.join(args.output_dir, 'concept', '{}.pt'.format(name)), 'wb') as f:
                for o in outputs:
                    print(o.shape)
                    pickle.dump(o, f)

        for name in ['train', 'dev']:
            dataset = []
            with open(os.path.join(args.output_dir, 'entity', '%s.json'%(name))) as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            outputs = encode_entity_dataset(args, vocab, dataset,tokenizer)
            assert len(outputs) == 6
            print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
            with open(os.path.join(args.output_dir, 'entity', '{}.pt'.format(name)), 'wb') as f:
                for o in outputs:
                    print(o.shape)
                    pickle.dump(o, f)

        for name in ['train', 'dev']:
            dataset = []
            with open(os.path.join(args.output_dir, 'attribute', '%s.json' % (name))) as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            outputs = encode_attribute_dataset(args, vocab, dataset,tokenizer)
            assert len(outputs) == 6
            print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
            with open(os.path.join(args.output_dir, 'attribute', '{}.pt'.format(name)), 'wb') as f:
                for o in outputs:
                    print(o.shape)
                    pickle.dump(o, f)

        for name in ['train', 'dev']:
            dataset = []
            with open(os.path.join(args.output_dir, 'operator', '%s.json' % (name))) as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            outputs = encode_operator_dataset(args, vocab, dataset,tokenizer)
            assert len(outputs) == 6
            print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
            with open(os.path.join(args.output_dir, 'operator', '{}.pt'.format(name)), 'wb') as f:
                for o in outputs:
                    print(o.shape)
                    pickle.dump(o, f)


        

if __name__ == "__main__":
    main()
