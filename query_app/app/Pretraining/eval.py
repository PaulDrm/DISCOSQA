import os
import argparse

from Pretraining.utils import *
from Pretraining.model import RelationPT
from Pretraining.model_rob import RelationPT_rob
from Pretraining.data import DataLoader
from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)
from Pretraining.lr_scheduler import get_linear_schedule_with_warmup
from Pretraining.metric import *

from transformers import (RobertaConfig, RobertaModel,AutoTokenizer, RobertaPreTrainedModel)

#from Pretraining.eval import evaluate
import torch
import torch.nn as nn
import json
from torch.utils.data import TensorDataset, SequentialSampler # DataLoader, RandomSampler, SequentialSampler
from pathlib import Path
from tqdm import tqdm
import pickle
import logging

import wandb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

def log_data(log):

    with open("./eval/results.json", 'r') as f:
        dataset = json.load(f)

    dataset.append(log)
    with open("./eval/results.json", 'w+') as f:
        json.dump(dataset, f)

def evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model, device, global_step=0,
             prefix='', **val_loaders):
    ## relation_eval_loader, concept_eval_loader, entity_eval_loader, attribute_eval_loader

    # eval_output_dir = args.output_dir
    # if not os.path.exists(eval_output_dir):
    #     os.makedirs(eval_output_dir)
    print(global_step)
    checkpoint =global_step
    ############################ Eval!
    ## Operators!
    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['operator_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['operator_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
    function_loss = 0
    for step, batch in enumerate(val_loaders['operator_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                # 'concept_inputs': concept_inputs,
                # 'relation_inputs': relation_inputs,
                # 'entity_inputs': entity_inputs,
                # 'attribute_inputs': attribute_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                # 'attribute_info': (batch[4], None),
                # 'relation_info': (batch[4], None),
                'concept_info': None,
                'entity_info': None,
                # 'entity_embeddings': None
                'operator_info': (batch[4], None)
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_operator']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['operator_logits'], gt_relation).item())
            function_loss += float(outputs['function_loss'].item())
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            # print(correct)
            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
        nb_eval_steps += 1
        pbar(step)
    logging.info('')
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    log = {'checkpoint':checkpoint, 'acc_operations': acc, "op_val_loss": val_loss, 'step': global_step}
    log_data(log)
    if args.wandb:
        wandb.log(log)
    logging.info('**** operation results %s ****', prefix)
    logging.info('acc: {}'.format(acc))

    # Eval!
    ## Attributes!
    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['attribute_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['attribute_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
    function_loss = 0
    for step, batch in enumerate(val_loaders['attribute_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                # 'concept_inputs': concept_inputs,
                # 'relation_inputs': relation_inputs,
                # 'entity_inputs': entity_inputs,
                'attribute_inputs': attribute_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'attribute_info': (batch[4], None),
                # 'relation_info': (batch[4], None),
                'concept_info': None,
                'entity_info': None,
                #   'entity_embeddings': None
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_attribute']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['attribute_logits'], gt_relation).item())
            function_loss += float(outputs['function_loss'].item())
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            # print(correct)



            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
        nb_eval_steps += 1
        pbar(step)
    logging.info('')
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    log = {'checkpoint': checkpoint, 'function_loss': function_loss/(step+1), 'acc_func': func_metric.result(), 'acc_attributes': acc, "att_val_loss": val_loss, 'step': global_step}
    log_data(log)
    if args.wandb:
        wandb.log(log)
    logging.info('**** attribute results %s ****', prefix)
    logging.info('acc: {}'.format(acc))

    ## Relations!
    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['relation_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['relation_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
    function_loss = 0
    for step, batch in enumerate(val_loaders['relation_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs,
                'relation_inputs': relation_inputs,
                'entity_inputs': entity_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': (batch[4], None),
                'concept_info': None,
                'entity_info': None,
                #   'entity_embeddings': None
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_relation']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['relation_logits'], gt_relation).item())
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            function_loss += float(outputs['function_loss'].item())
            # print(correct)
            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
        nb_eval_steps += 1
        pbar(step)
    logging.info('')
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    log = {'checkpoint':checkpoint,'function_loss': function_loss/(step+1), 'acc_func': func_metric.result(), 'acc_relations': acc, "rel_val_loss": val_loss, 'step': global_step}
    log_data(log)
    if args.wandb:
       wandb.log(log)
    logging.info('**** relation results %s ****', prefix)
    logging.info('acc: {}'.format(acc))

    ## Concepts!
    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['concept_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['concept_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
    function_loss = 0
    for step, batch in enumerate(val_loaders['concept_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs,
                'relation_inputs': relation_inputs,
                'entity_inputs': entity_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': None,
                'concept_info': (batch[4], None),
                'entity_info': None,
                #  'entity_embeddings': None
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_concept']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['concept_logits'], gt_relation).item())
            function_loss += float(outputs['function_loss'].item())
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            # print(correct)
            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
        nb_eval_steps += 1
        pbar(step)
    logging.info('')
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    logging.info('**** concept results %s ****', prefix)
    logging.info('acc: {}'.format(acc))
    log = {'checkpoint':checkpoint, 'function_loss': function_loss/(step+1), 'acc_func': func_metric.result(), 'acc_concepts': acc, "cons_val_loss": val_loss, 'step': global_step}
    log_data(log)
    if args.wandb:
        wandb.log(log)

    # Entities!
    # with torch.no_grad():
    #     model.entity_embeddings = model.bert(input_ids=entity_inputs['input_ids'],
    #                                     attention_mask=entity_inputs['attention_mask'],
    #                                     token_type_ids=entity_inputs['token_type_ids'])[1]

    # with open(os.path.abspath(args.input_dir + "/entity/entity_embeddings_3110.pt"), 'rb') as f:

    #    model.entity_embeddings = pickle.load(f)
    # with open('c_embeddings.pt', 'wb') as f: #os.path.join(args.output_dir,
    #           # for o in concept_embeddings:
    #            # print(o)
    #   pickle.dump(concept_embeddings, f)

    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['entity_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['entity_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
    results = []
    function_loss = 0

    for step, batch in enumerate(val_loaders['entity_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())

        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs,
                'relation_inputs': relation_inputs,
                'entity_inputs': entity_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': None,
                'concept_info': None,
                'entity_info': (batch[4], None),
                # 'entity_embeddings':entity_embeddings
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_entity']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['entity_logits'], gt_relation).item())
            # print(pred_relation.size(), gt_relation.size(), batch[3].size())
            function_loss += float(outputs['function_loss'].item())
            correct += torch.sum(torch.eq(pred_relation, gt_relation).float())
            # print(correct)
            tot += len(pred_relation)
            gt_functions = batch[3].cpu().tolist()
            for pred, gt in zip(pred_functions, gt_functions):
                func_metric.update(pred, gt)
            end_id = val_loaders['entity_val_loader'].vocab['function2id']['<END>']
            boolean = []
            for pred, label in zip(pred_functions, gt_functions):
                for i in range(min(len(pred), len(label))):
                    if label[i] != pred[i]:
                        match = False
                        boolean.append(True)
                        break
                    if pred[i] == end_id and label[i] == end_id:
                        boolean.append(False)
                        break
            if args.model_type == 'roberta':
                tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            sents = tokenizer.batch_decode(inputs['input_ids'].to('cpu'))

            for des, val, pred, label in zip(boolean, sents, pred_functions, gt_functions):
                results.append({'question': val, 'prediction': pred, 'functions': label, 'label': des})
        nb_eval_steps += 1
        pbar(step)
    logging.info('')
    acc = func_metric.result()
    logging.info('**** function results %s ****', prefix)
    info = 'acc: {}'.format(acc)
    logging.info(info)
    acc = correct.item() / tot
    logging.info('**** entity results %s ****', prefix)
    logging.info('acc: {}'.format(acc))
    log = {'checkpoint':checkpoint, 'function_loss': function_loss/(step+1), 'acc_func': func_metric.result(), 'acc_entities': acc, "ent_val_loss": val_loss, 'step': global_step}
    print(results[0:3])
    with open(f'./eval/function_predictions_{checkpoint}.json', 'w') as fp:
        json.dump(results, fp)
    log_data(log)

    if args.wandb:
        wandb.log(log)




def embed_ents(model,args):
    batch_num = 128
    # argument_inputs = load_classes(input_dir + "esa/new/entity_3110.pt", )
    argument_inputs = load_classes(args.data_dir + "/entity/entity.pt",'cpu')
    data = TensorDataset(argument_inputs['input_ids'], argument_inputs['attention_mask'],
                         argument_inputs['token_type_ids'])
    data_sampler = SequentialSampler(data)
    dataloader = torch.utils.data.DataLoader(data, sampler=data_sampler, batch_size=batch_num)

    attribute_embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # if i == 1:
            #    break
            inputs = batch[0].to(device)
            masks = batch[1].to(device)
            tags = batch[2].to(device)

            attribute_embeddings += model.bert(input_ids=inputs,
                                               attention_mask=masks,
                                               token_type_ids=tags)[1].cpu()
    attribute_embeddings = torch.stack(attribute_embeddings)

    model.entity_embeddings = attribute_embeddings.cuda()

def load_classes(path,device):
  with open(os.path.abspath(path), 'rb') as f:
      input_ids = pickle.load(f)
      token_type_ids = pickle.load(f)
      attention_mask = pickle.load(f)
      # input_ids = torch.LongTensor(input_ids[:512,:]).to(device)
      # token_type_ids = torch.LongTensor(token_type_ids[:512,:]).to(device)
      # attention_mask = torch.LongTensor(attention_mask[:512,:]).to(device)
      input_ids = torch.LongTensor(input_ids).to(device)
      token_type_ids = torch.LongTensor(token_type_ids).to(device)
      attention_mask = torch.LongTensor(attention_mask).to(device)
  argument_inputs = {
    'input_ids': input_ids,
    'token_type_ids': token_type_ids,
    'attention_mask': attention_mask
  }
  return argument_inputs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default='./train/')
    parser.add_argument('--data_dir', type=str, default='./test_data/')
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--model_type', type=str, default="bert")
    #parser.add_argument('--save_dir', type=int, default=128)

    args = parser.parse_args()
    vocab_json = os.path.join(args.data_dir, 'vocab.json')
    operator_val_pt = os.path.join(args.data_dir, 'operator', 'dev.pt')
    concept_val_pt = os.path.join(args.data_dir, 'concept', 'dev.pt')
    entity_val_pt = os.path.join(args.data_dir, 'entity', 'dev.pt')
    relation_val_pt = os.path.join(args.data_dir, 'relation', 'dev.pt')
    attribute_val_pt = os.path.join(args.data_dir, 'attribute', 'dev.pt')

    concept_val_loader = DataLoader(vocab_json, concept_val_pt, args.val_batch_size)
    relation_val_loader = DataLoader(vocab_json, relation_val_pt, args.val_batch_size)
    entity_val_loader = DataLoader(vocab_json, entity_val_pt, args.val_batch_size)
    attribute_val_loader = DataLoader(vocab_json, attribute_val_pt, args.val_batch_size)
    operator_val_loader = DataLoader(vocab_json, operator_val_pt, args.val_batch_size)

    val_loaders = {'entity_val_loader': entity_val_loader,
                   'concept_val_loader': concept_val_loader,
                   'attribute_val_loader': attribute_val_loader,
                   'operator_val_loader': operator_val_loader,
                   'relation_val_loader': relation_val_loader,
                   }
    models_path = Path(args.models_dir)

    if args.wandb:
        wandb.init(project="ProgramTransfer_Augmentation1001", name=args.models_dir)

    for model_dir in models_path.iterdir():
        if not model_dir.is_dir():
            continue

        if args.model_type == 'bert':

            config_class, model_class = (BertConfig, RelationPT)

        else:

            config_class, model_class = (RobertaConfig, RelationPT_rob)

        print("load ckpt from {}".format(model_dir))
        config = config_class.from_pretrained(model_dir)  # , num_labels = len(label_list))
        model = model_class.from_pretrained(model_dir, config=config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():  #
            model.cuda()

        embed_ents(model,args)

        attribute_inputs = load_classes(args.data_dir + "attribute/attribute.pt", device)
        # with torch.no_grad():
        #     attribute_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
        #                                       attention_mask=argument_inputs['attention_mask'],
        #                                       token_type_ids=argument_inputs['token_type_ids'])[1]

        concept_inputs = load_classes(args.data_dir + "concept/concept.pt", device)
        # with torch.no_grad():
        #     concept_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
        #                                     attention_mask=argument_inputs['attention_mask'],
        #                                     token_type_ids=argument_inputs['token_type_ids'])[1]

        ##argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
        # with torch.no_grad():
        #     relation_embeddings = _model.bert(input_ids=argument_inputs['input_ids'],
        #                                      attention_mask=argument_inputs['attention_mask'],
        #                                      token_type_ids=argument_inputs['token_type_ids'])[1]

        relation_inputs = load_classes(args.data_dir + "relation/relation.pt", device)
        # with torch.no_grad():
        #     relation_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
        #                                      attention_mask=argument_inputs['attention_mask'],
        #                                      token_type_ids=argument_inputs['token_type_ids'])[1]

        #checkpoint = str(model_dir).split('\\')[-1]
        checkpoint = str(model_dir).split("checkpoint-")[1]
        entity_inputs = []
        evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,device, global_step=int(checkpoint), **val_loaders)

if __name__ == '__main__':
    main()








