import os
import torch

import argparse
import shutil
#from tqdm import tqdm
#import numpy as np
from Pretraining.utils import *
from Pretraining.model import RelationPT
from Pretraining.data import DataLoader
from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)
from Pretraining.lr_scheduler import get_linear_schedule_with_warmup
from Pretraining.metric import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import torch.optim as optim
import torch.nn as nn
import wandb
import sys
#from IPython import embed
import pickle
os.environ["WANDB_API_KEY"] = "7ee09b0cec0f14411947fcf09144f4a72c09c411"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model, device,global_step=0, prefix = '',**val_loaders):

    ## relation_eval_loader, concept_eval_loader, entity_eval_loader, attribute_eval_loader

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    ############################ Eval!
    ## Operators!
    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['operator_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['operator_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
    for step, batch in enumerate(val_loaders['operator_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                #'concept_inputs': concept_inputs,
                #'relation_inputs': relation_inputs,
                #'entity_inputs': entity_inputs,
                #'attribute_inputs': attribute_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                #'attribute_info': (batch[4], None),
                #'relation_info': (batch[4], None),
                'concept_info': None,
                'entity_info': None,
                #'entity_embeddings': None
                'operator_info': (batch[4], None)
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_operator']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['operator_logits'], gt_relation).item())
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
    log = {'acc_func': func_metric.result(), 'acc_operations': acc, "op_val_loss":val_loss, 'step': global_step}

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
    for step, batch in enumerate(val_loaders['attribute_val_loader']):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        # print(batch[4].size())
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                #'concept_inputs': concept_inputs,
                #'relation_inputs': relation_inputs,
                #'entity_inputs': entity_inputs,
                'attribute_inputs': attribute_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'attribute_info':(batch[4], None),
                #'relation_info': (batch[4], None),
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
    log = {'acc_func': func_metric.result(), 'acc_attributes': acc,"att_val_loss":val_loss, 'step': global_step}
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
    log = {'acc_func': func_metric.result(), 'acc_relations': acc,"rel_val_loss":val_loss, 'step': global_step}
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
                'entity_info':  None,
             #  'entity_embeddings': None
            }
            outputs = model(**inputs)
            pred_functions = outputs['pred_functions'].cpu().tolist()
            pred_relation = outputs['pred_concept']
            gt_relation = batch[5]
            gt_relation = gt_relation.squeeze(-1)
            val_loss += float(nn.CrossEntropyLoss()(outputs['concept_logits'], gt_relation).item())
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
    log = {'acc_func': func_metric.result(), 'acc_concepts':acc , "cons_val_loss":val_loss, 'step': global_step}
    if args.wandb:
        wandb.log(log)

    # Entities!
    # with torch.no_grad():
    #     model.entity_embeddings = model.bert(input_ids=entity_inputs['input_ids'],
    #                                     attention_mask=entity_inputs['attention_mask'],
    #                                     token_type_ids=entity_inputs['token_type_ids'])[1]

    #with open(os.path.abspath(args.input_dir + "/entity/entity_embeddings_3110.pt"), 'rb') as f:

    #    model.entity_embeddings = pickle.load(f)
    #with open('c_embeddings.pt', 'wb') as f: #os.path.join(args.output_dir,
    #           # for o in concept_embeddings:
    #            # print(o)
    #   pickle.dump(concept_embeddings, f)

    nb_eval_steps = 0
    func_metric = FunctionAcc(val_loaders['entity_val_loader'].vocab['function2id']['<END>'])
    pbar = ProgressBar(n_total=len(val_loaders['entity_val_loader']), desc="Evaluating")
    correct = 0
    tot = 0
    val_loss = 0
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
    logging.info('**** entity results %s ****', prefix)
    logging.info('acc: {}'.format(acc))
    log = {'acc_func': func_metric.result(), 'acc_entities': acc, "ent_val_loss":val_loss, 'step': global_step}
    if args.wandb:
        wandb.log(log)

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    ## breaks loops for test run case

    test = 0
    #print(args.batch_size)
    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    relation_train_pt = os.path.join(args.input_dir, 'relation', 'train.pt')
    relation_val_pt = os.path.join(args.input_dir, 'relation', 'dev.pt')
    relation_train_loader = DataLoader(vocab_json, relation_train_pt, args.train_batch_size, training=True)
    relation_val_loader = DataLoader(vocab_json, relation_val_pt, args.val_batch_size)

    concept_train_pt = os.path.join(args.input_dir, 'concept', 'train.pt')
    concept_val_pt = os.path.join(args.input_dir, 'concept', 'dev.pt')
    concept_train_loader = DataLoader(vocab_json, concept_train_pt, args.train_batch_size, training=True)
    concept_val_loader = DataLoader(vocab_json, concept_val_pt, args.val_batch_size)

    entity_train_pt = os.path.join(args.input_dir, 'entity', 'train.pt')
    entity_val_pt = os.path.join(args.input_dir, 'entity', 'dev.pt')
    entity_train_loader = DataLoader(vocab_json, entity_train_pt, args.train_batch_size, training=True)
    entity_val_loader = DataLoader(vocab_json, entity_val_pt, args.val_batch_size)
    #entity_train_loader = DataLoader(vocab_json, concept_train_pt, args.batch_size, training=True)
    #entity_val_loader = DataLoader(vocab_json, concept_val_pt, args.batch_size)

    attribute_train_pt = os.path.join(args.input_dir, 'attribute', 'train.pt')
    attribute_val_pt = os.path.join(args.input_dir, 'attribute', 'dev.pt')
    attribute_train_loader = DataLoader(vocab_json, attribute_train_pt, args.train_batch_size, training=True)
    attribute_val_loader = DataLoader(vocab_json, attribute_val_pt, args.val_batch_size)
    # entity_train_loader = DataLoader(vocab_json, concept_train_pt, args.batch_size, training=True)
    # entity_val_loader = DataLoader(vocab_json, concept_val_pt, args.batch_size)

    operator_train_pt = os.path.join(args.input_dir, 'operator', 'train.pt')
    operator_val_pt = os.path.join(args.input_dir, 'operator', 'dev.pt')
    operator_train_loader = DataLoader(vocab_json, operator_train_pt, args.train_batch_size, training=True)
    operator_val_loader = DataLoader(vocab_json, operator_val_pt, args.val_batch_size)

    val_loaders = {'entity_val_loader': entity_val_loader,
                   'concept_val_loader': concept_val_loader,
                   'attribute_val_loader': attribute_val_loader,
                   'operator_val_loader': operator_val_loader,
                   'relation_val_loader':relation_val_loader,
    }

    with open(os.path.join(args.input_dir, 'relation', 'relation.pt'), 'rb') as f:
        input_ids = pickle.load(f)
        token_type_ids = pickle.load(f)
        attention_mask = pickle.load(f)
        input_ids = torch.LongTensor(input_ids).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
    relation_inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    with open(os.path.join(args.input_dir, 'concept', 'concept.pt'), 'rb') as f:
        input_ids = pickle.load(f)
        token_type_ids = pickle.load(f)
        attention_mask = pickle.load(f)
        input_ids = torch.LongTensor(input_ids).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
    concept_inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    with open(os.path.join(args.input_dir, 'attribute', 'attribute.pt'), 'rb') as f:
        input_ids = pickle.load(f)
        token_type_ids = pickle.load(f)
        attention_mask = pickle.load(f)
        input_ids = torch.LongTensor(input_ids).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
    attribute_inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    with open(os.path.join(args.input_dir, 'entity', 'entity.pt'), 'rb') as f:
        entities_input_ids = pickle.load(f)
        entities_token_type_ids = pickle.load(f)
        entities_attention_mask = pickle.load(f)
        input_ids = torch.LongTensor(entities_input_ids).to(device)
        token_type_ids = torch.LongTensor(entities_token_type_ids).to(device)
        attention_mask = torch.LongTensor(entities_attention_mask).to(device)
    entity_inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    vocab = relation_train_loader.vocab
    
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BertConfig, RelationPT, BertTokenizer)
    config = config_class.from_pretrained(args.model_name_or_path, num_labels = len(label_list))
    config.update({'vocab': vocab})
    tokenizer = tokenizer_class.from_pretrained("bert-base-cased", do_lower_case = False)

    model = model_class.from_pretrained(args.model_name_or_path, config = config)
    model = model.to(device)
    # logging.info(model)

    ## # Prepare optimizer and schedule (linear warmup and decay)
    t_total = (len(relation_train_loader) + len(concept_train_loader) + len(entity_train_loader) + len(attribute_train_loader)+ len(operator_train_loader))// args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())

    ## Number of optimization parameters
    linear_param_optimizer =   list(model.function_embeddings.named_parameters()) \
                             + list(model.function_classifier.named_parameters()) \
                             + list(model.function_decoder.named_parameters()) \
                             + list(model.relation_classifier.named_parameters())\
                             + list(model.concept_classifier.named_parameters())\
                             + list(model.entity_classifier.named_parameters())\
                             + list(model.attribute_classifier.named_parameters())\
                             + list(model.operation_classifier.named_parameters())
    ## Linear warmup and decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(relation_train_loader.dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(relation_train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(relation_train_loader) // args.gradient_accumulation_steps)
        logging.info("  Continuing training from checkpoint, will skip to saved global_step")
        logging.info("  Continuing training from epoch %d", epochs_trained)
        logging.info("  Continuing training from global step %d", global_step)
        logging.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    logging.info('Checking...')
    logging.info("===================Dev==================")
    # evaluate(args, concept_inputs, relation_inputs, model, relation_val_loader, concept_val_loader, device)
    tot_tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    evaluation =1
    if evaluation:
        model.eval()
        with torch.no_grad():
            evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,
                     device, global_step, **val_loaders)
        sys.exit(0)

    for _ in range(int(args.num_train_epochs)):

        model.train()

        logging.info('Operator training begins')
        pbar = ProgressBar(n_total=len(operator_train_loader), desc='Training')
        for step, batch in enumerate(operator_train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                # 'concept_inputs': concept_inputs,
                # 'relation_inputs': relation_inputs,
                # 'entity_inputs': entity_inputs,
                #'attribute_inputs': attribute_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                #'attribute_info': (batch[4], batch[5])
                'operator_info': (batch[4], batch[5]),
                # 'relation_info': (batch[4], batch[5]),
                # 'concept_info': None,
                # 'entity_info': None,
                # 'entity_embeddings': None
            }

            outputs = model(**inputs)
            loss = args.rel * outputs['operator_loss']
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tot_tr_loss += float(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1

            if args.wandb and (global_step + 1) % args.logging_steps == 0:
                wandb.log({"train_loss_tot": tot_tr_loss / global_step, "epoch": _, "step": global_step,
                           'lr_BERT': scheduler.get_last_lr()[0], 'lr_class': scheduler.get_last_lr()[2]})

            if (global_step + 1) % args.eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,
                             device, global_step, **val_loaders)
                wandb.log({"train_loss_tot": tot_tr_loss / global_step, "epoch": _, "step": global_step,
                           'lr_BERT': scheduler.get_last_lr()[0], 'lr_class': scheduler.get_last_lr()[2]})

                model.train()
            if (global_step + 1) % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)
                tokenizer.save_vocabulary(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logging.info("Saving optimizer and scheduler states to %s", output_dir)
                logging.info("\n")

            if test:
                print("Testcase finished successfully")
                break


        logging.info('Attribute training begins')
        pbar = ProgressBar(n_total=len(attribute_train_loader), desc='Training')
        for step, batch in enumerate(attribute_train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                #'concept_inputs': concept_inputs,
                #'relation_inputs': relation_inputs,
                #'entity_inputs': entity_inputs,
                'attribute_inputs': attribute_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'attribute_info': (batch[4], batch[5])
               # 'relation_info': (batch[4], batch[5]),
               # 'concept_info': None,
               # 'entity_info': None,
               # 'entity_embeddings': None
            }
            outputs = model(**inputs)
            loss = args.func * outputs['function_loss'] + args.rel * outputs['attribute_loss']
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tot_tr_loss += float(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1

            ## WANDBDB Logging
            if (global_step+ 1) % args.logging_steps == 0:
                wandb.log({"train_loss_tot": tot_tr_loss/global_step, "epoch": _, "step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})
            ## EVALUATION
            if (global_step + 1) % args.eval_steps == 0:
                wandb.log({"train_loss_tot": tot_tr_loss / global_step, "epoch": _,"step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})
                model.eval()
                with torch.no_grad():
                    #evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model, relation_val_loader, concept_val_loader, entity_val_loader, attribute_val_loader, device,global_step)
                    evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,
                             device, global_step, **val_loaders)
                model.train()
            if (global_step+1) % args.save_steps== 0:
                 # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)
                tokenizer.save_vocabulary(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logging.info("Saving optimizer and scheduler states to %s", output_dir)
                logging.info("\n")

            if test:
                print("Testcase finished successfully")
                break

        logging.info('entity training begins')
        pbar = ProgressBar(n_total=len(entity_train_loader), desc='Training')
        entity_loss = 0
        epoch_step = 0
        epoch_tr_loss = 0
        for step, batch in enumerate(entity_train_loader):
            
            # if step == 0:
            #     print('Calculating embeddings for step {}...'.format(step))
            #     with torch.no_grad():
            #         entity_embeddings = model.bert(input_ids=entity_inputs['input_ids'],
            #                                         attention_mask=entity_inputs['attention_mask'],
            #                                         token_type_ids=entity_inputs['token_type_ids'])[1]
            #     #with open('c_embeddings.pt', 'wb') as f: #os.path.join(args.output_dir,
            #        # for o in concept_embeddings:
            #         # print(o)
            #     #    pickle.dump(concept_embeddings, f)
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            ## [[6124], [10234], [5], [5], [1]] -> [6124, 10234, 5, 5, 1]
            index = [index[0].cpu().item() for index in batch[5]]
            #  [6124, 10234, 5, 5, 1] -> [9008,8008,1273,...,6124,10234,5,5,1]
            index = torch.randint(len(entities_input_ids), (512-args.train_batch_size,)).tolist()+index
            #  [9008,8008,1273,...,6124,10234,5,5,1] -> [9008,8008,1273,...,6124,10234,5,1]
            index = list(set(index))
            #print(index)
            input_ids = torch.LongTensor(np.array([entities_input_ids[i] for i in index])).to(device)  ## np.array increases processing speed, bug python
            token_type_ids = torch.LongTensor(np.array([entities_token_type_ids[i] for i in index])).to(device)
            attention_mask = torch.LongTensor(np.array([entities_attention_mask[i] for i in index])).to(device)

            batch_entity_inputs = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask
            }

            batch_pos = [[index.index(i)] for i in batch[5].squeeze(-1)]
            batch_pos = torch.LongTensor(batch_pos).to(device)
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': None,
                'relation_inputs': None,
                'entity_inputs': batch_entity_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': None,
                'concept_info': None,
                'entity_info': (batch[4], batch_pos),
            #    'entity_embeddings':None
            }
            outputs = model(**inputs)
            loss = args.func * outputs['function_loss'] + args.con * outputs['entity_loss']
            loss.backward()

            entity_loss += float(outputs['entity_loss'].item())

            pbar(step, {'loss': loss.item()})
            epoch_tr_loss += float(loss.item())
            tot_tr_loss += float(loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                
                #print('Calculating embeddings for step {}...'.format(step))
                #with torch.no_grad():
                #    entity_embeddings = model.bert(input_ids=entity_inputs['input_ids'],
                #                                    attention_mask=entity_inputs['attention_mask'],
                #                                    token_type_ids=entity_inputs['token_type_ids'])[1]
                
            global_step += 1
            epoch_step+=1
            if (global_step+ 1) % args.logging_steps == 0:
                wandb.log({"train_loss_tot": tot_tr_loss/global_step,"train_loss_epoch":epoch_tr_loss/epoch_step, "train_loss_ent": entity_loss/epoch_step, "epoch": _, "step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})

            if (global_step + 1) % args.eval_steps == 0:

                model.eval()
                with torch.no_grad():
                    #evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model, relation_val_loader, concept_val_loader, entity_val_loader, attribute_val_loader, device,global_step)
                    evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,
                             device, global_step, **val_loaders)
                model.train()
                wandb.log({"train_loss_tot": tot_tr_loss / global_step,"train_loss_epoch":epoch_tr_loss/epoch_step, "train_loss_ent": entity_loss/epoch_step,"epoch": _, "step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})
            if (global_step+1) % args.save_steps== 0:
                 # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)
                tokenizer.save_vocabulary(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logging.info("Saving optimizer and scheduler states to %s", output_dir)
                logging.info("\n")


            if test:
                break

        logging.info('relation training begins')
        pbar = ProgressBar(n_total=len(relation_train_loader), desc='Training')
        for step, batch in enumerate(relation_train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'concept_inputs': concept_inputs,
                'relation_inputs': relation_inputs,
                'entity_inputs': entity_inputs,
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
                'function_ids': batch[3],
                'relation_info': (batch[4], batch[5]),
                'concept_info': None,
                'entity_info': None,
             #   'entity_embeddings': None
            }
            outputs = model(**inputs)
            loss = args.func * outputs['function_loss'] + args.rel * outputs['relation_loss']
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tot_tr_loss += float(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1

            if (global_step+ 1) % args.logging_steps == 0:
                wandb.log({"train_loss_tot": tot_tr_loss/global_step, "epoch": _, "step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})

            if (global_step + 1) % args.eval_steps == 0:
                wandb.log({"train_loss_tot": tot_tr_loss / global_step, "epoch": _,"step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})
                model.eval()
                with torch.no_grad():
                    #evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model, relation_val_loader, concept_val_loader, entity_val_loader, attribute_val_loader, device,global_step)
                    evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,
                             device, global_step, **val_loaders)
                model.train()
            if (global_step+1) % args.save_steps== 0:
                 # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)
                tokenizer.save_vocabulary(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logging.info("Saving optimizer and scheduler states to %s", output_dir)
                logging.info("\n")

            if test:
                break

        logging.info('concept training begins')
        pbar = ProgressBar(n_total=len(concept_train_loader), desc='Training')
        for step, batch in enumerate(concept_train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
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
                'concept_info': (batch[4], batch[5]),
                'entity_info': None,
            #    'entity_embeddings': None
            }
            outputs = model(**inputs)
            loss = args.func * outputs['function_loss'] + args.con * outputs['concept_loss']
            loss.backward()
            pbar(step, {'loss': loss.item()})
            #tr_loss += loss.item()
            tot_tr_loss+= float(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1

            if (global_step+ 1) % args.logging_steps == 0:
                wandb.log({"train_loss_tot":  tot_tr_loss/global_step, "epoch": _, "step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})

            if (global_step + 1) % args.eval_steps == 0:
                wandb.log({"train_loss_tot":  tot_tr_loss / global_step, "epoch": _, "step": global_step, 'lr_BERT':scheduler.get_last_lr()[0], 'lr_class':scheduler.get_last_lr()[2]})
                model.eval()
                with torch.no_grad():
                    #evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model, relation_val_loader, concept_val_loader, entity_val_loader, attribute_val_loader, device,global_step)
                    evaluate(args, concept_inputs, relation_inputs, entity_inputs, attribute_inputs, model,
                             device, global_step, **val_loaders)
                model.train()
            if (global_step+1) % args.save_steps== 0:
                 # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)
                tokenizer.save_vocabulary(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logging.info("Saving optimizer and scheduler states to %s", output_dir)
                logging.info("\n")
            if test:
                break

        if 'cuda' in str(device):
            torch.cuda.empty_cache()


        #with torch.no_grad():
         #   evaluate(args, concept_inputs, relation_inputs, entity_inputs, model, relation_val_loader, concept_val_loader,entity_val_loader, device)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    # parser.add_argument('--glove_pt', default='/data/csl/resources/word2vec/glove.840B.300d.py36.pt')
    # parser.add_argument('--model_name_or_path', default = '/data/csl/resources/Bert/bert-base-cased')
    parser.add_argument('--model_name_or_path', default='bert-base-cased')

    # parser.add_argument('--ckpt')
    parser.add_argument('--wandb', default=1, type=int)
    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type = float)
    parser.add_argument('--crf_learning_rate', default=1e-3, type = float)
    parser.add_argument('--num_train_epochs', default=25, type = int)
    parser.add_argument('--save_steps', default=2400, type = int)
    parser.add_argument("--eval_steps",default=2000, type=int)
    parser.add_argument('--logging_steps', default=400, type = int)
    parser.add_argument('--warmup_proportion', default=0.1, type = float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1, type = float)
    parser.add_argument('--beta', default = 1e-1, type = float)
    parser.add_argument('--func', default = 1, type = float)
    parser.add_argument('--rel', default = 1, type = float)
    parser.add_argument('--con', default = 1, type = float)


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.time()#strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    seed_everything(666)
    if args.wandb:
        wandb.init(project="ProgramTransfer_test")
    train(args)

if __name__ == '__main__':
    main()
