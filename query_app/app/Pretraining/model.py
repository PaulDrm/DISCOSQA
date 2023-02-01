import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import *
from transformers import (BertConfig, BertModel,  BertPreTrainedModel)
from transformers import (AutoConfig, AutoModel, RobertaPreTrainedModel)
from Pretraining.utils import *
from Pretraining.BiGRU import GRU, BiGRU
import pickle
import re
import numpy as np
import copy

from sys import implementation
import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, dim_word, dim_h, num_layers, dropout):
        super().__init__()
        self.encoder = nn.GRU(input_size=dim_word,
                              hidden_size=dim_h,
                              num_layers=num_layers,
                              dropout=dropout,
                              batch_first=True,
                              bidirectional=False)

    def forward_one_step(self, input, last_h):
        """
        Args:
            - input (bsz, 1, w_dim)
            - last_h (num_layers, bsz, h_dim)
        """
        hidden, new_h = self.encoder(input, last_h)
        return hidden, new_h  # (bsz, 1, h_dim), (num_layers, bsz, h_dim)

    def generate_sequence(self, word_lookup_func, h_0, classifier, vocab, max_step, early_stop=True):
        bsz = h_0.size(1)
        device = h_0.device
        start_id, end_id, pad_id = vocab['<START>'], vocab['<END>'], vocab['<PAD>']

        latest = torch.LongTensor([start_id] * bsz).to(device)  # [bsz, ]
        results = [latest]
        last_h = h_0
        finished = torch.zeros((bsz,)).bool().to(device)  # record whether <END> is produced
        for i in range(max_step - 1):  # exclude <START>
            word_emb = word_lookup_func(latest).unsqueeze(1)  # [bsz, 1, dim_w]
            word_h, last_h = self.forward_one_step(word_emb, last_h)  # [bsz, 1, dim_h]

            logit = classifier(word_h).squeeze(1)  # [bsz, num_func]
            latest = torch.argmax(logit, dim=1).long()  # [bsz, ]
            latest[finished] = pad_id  # set to <PAD> after <END>
            results.append(latest)

            finished = finished | latest.eq(end_id).bool()
            if early_stop and finished.sum().item() == bsz:
                # print('finished at step {}'.format(i))
                break
        results = torch.stack(results, dim=1)  # [bsz, max_len']
        return results

    def forward(self, input, length, h_0=None):
        """
        Args:
            - input (bsz, len, w_dim)
            - length (bsz, )
            - h_0 (num_layers, bsz, h_dim)
        Return:
            - hidden (bsz, len, dim) : hidden state of each word
            - output (bsz, dim) : sentence embedding
        """
        bsz, max_len = input.size(0), input.size(1)
        sorted_seq_lengths, indices = torch.sort(length, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        input = input[indices]
        packed_input = nn.utils.rnn.pack_padded_sequence(input, sorted_seq_lengths, batch_first=True)
        if h_0 is None:
            hidden, h_n = self.encoder(packed_input)
        else:
            h_0 = h_0[:, indices]
            hidden, h_n = self.encoder(packed_input, h_0)
        # h_n is (num_layers, bsz, h_dim)
        hidden = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=max_len)[
            0]  # (bsz, max_len, h_dim)

        output = h_n[-1, :, :]  # (bsz, h_dim), take the last layer's state

        # recover order
        hidden = hidden[desorted_indices]
        output = output[desorted_indices]
        h_n = h_n[:, desorted_indices]
        return hidden, output, h_n


import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, dim_word, dim_h, num_layers, dropout):
        super().__init__()
        self.encoder = nn.GRU(input_size=dim_word,
                              hidden_size=dim_h,
                              num_layers=num_layers,
                              dropout=dropout,
                              batch_first=True,
                              bidirectional=False)

    def forward_one_step(self, input, last_h):
        """
        Args:
            - input (bsz, 1, w_dim)
            - last_h (num_layers, bsz, h_dim)
        """
        hidden, new_h = self.encoder(input, last_h)
        return hidden, new_h  # (bsz, 1, h_dim), (num_layers, bsz, h_dim)

    def generate_sequence(self, word_lookup_func, h_0, classifier, vocab, max_step, early_stop=True):
        bsz = h_0.size(1)
        device = h_0.device
        start_id, end_id, pad_id = vocab['<START>'], vocab['<END>'], vocab['<PAD>']

        latest = torch.LongTensor([start_id] * bsz).to(device)  # [bsz, ]
        results = [latest]
        last_h = h_0
        finished = torch.zeros((bsz,)).bool().to(device)  # record whether <END> is produced
        for i in range(max_step - 1):  # exclude <START>
            word_emb = word_lookup_func(latest).unsqueeze(1)  # [bsz, 1, dim_w]
            word_h, last_h = self.forward_one_step(word_emb, last_h)  # [bsz, 1, dim_h]

            logit = classifier(word_h).squeeze(1)  # [bsz, num_func]
            latest = torch.argmax(logit, dim=1).long()  # [bsz, ]
            latest[finished] = pad_id  # set to <PAD> after <END>
            results.append(latest)

            finished = finished | latest.eq(end_id).bool()
            if early_stop and finished.sum().item() == bsz:
                # print('finished at step {}'.format(i))
                break
        results = torch.stack(results, dim=1)  # [bsz, max_len']
        return results

    def forward(self, input, length, h_0=None):
        """
        Args:
            - input (bsz, len, w_dim)
            - length (bsz, )
            - h_0 (num_layers, bsz, h_dim)
        Return:
            - hidden (bsz, len, dim) : hidden state of each word
            - output (bsz, dim) : sentence embedding
        """
        bsz, max_len = input.size(0), input.size(1)
        sorted_seq_lengths, indices = torch.sort(length, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        input = input[indices]
        packed_input = nn.utils.rnn.pack_padded_sequence(input, sorted_seq_lengths, batch_first=True)
        if h_0 is None:
            hidden, h_n = self.encoder(packed_input)
        else:
            h_0 = h_0[:, indices]
            hidden, h_n = self.encoder(packed_input, h_0)
        # h_n is (num_layers, bsz, h_dim)
        hidden = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=max_len)[
            0]  # (bsz, max_len, h_dim)

        output = h_n[-1, :, :]  # (bsz, h_dim), take the last layer's state

        # recover order
        hidden = hidden[desorted_indices]
        output = output[desorted_indices]
        h_n = h_n[:, desorted_indices]
        return hidden, output, h_n


from sys import implementation
import torch
# from transformers import AutoModelForCausalLM
# from transformers import AutoTokenizer
#from transformers import (BertModel, BertPreTrainedModel)


class RelationPT(BertPreTrainedModel):
    def __init__(self, config):

        ## Save large embeddings for fast prediction
        self.entity_embeddings = None

        super(RelationPT, self).__init__(config)
        self.vocab = config.vocab
        self.bert = BertModel(config)#AutoModel(config)#
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_functions = len(config.vocab['function2id'])
        self.function_embeddings = nn.Embedding(self.num_functions, config.hidden_size)
        #self.function_decoder = GRU(config.hidden_size, config.hidden_size, num_layers=1, dropout=0.2)
        self.function_decoder = GRU(config.hidden_size, config.hidden_size, num_layers=config.dec_hidden_layers, dropout=0.2)

        self.function_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_functions),
        )
        self.word_dropout = nn.Dropout(0.2)
        self.max_program_len = 17

        # self.relation_inputs = config.relation_inputs

        self.relation_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )

        self.concept_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )
        # ## Todo classifier changed
        self.entity_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )

        # ## Todo classifier changed
        self.attribute_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )
        # self.values_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, config.hidden_size),
        # )

        self.operation_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 5),
        )

        self.hidden_size = config.hidden_size
        self.init_weights()

    def demo(self, input_ids, token_type_ids, attention_mask, relation_embeddings=None, attribute_embeddings=None,
             concept_embeddings=None):

        with torch.no_grad():
            # input_ids = inputs['input_ids']
            # attention_mask = inputs['attention_mask']
            # token_type_ids = inputs['token_type_ids']

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = outputs[0]  # [bsz, max_seq_length, hidden_size]
            pooler_output = outputs[1]  # [bsz, hidden_size]
            outputs = {}
            sequence_output = self.dropout(sequence_output)
            bsz = input_ids.size(0)
            device = input_ids.device
            start_id = self.vocab['function2id']['<START>']
            end_id = self.vocab['function2id']['<END>']
            finished = torch.zeros((bsz,)).byte().to(device)  # record whether <END> is produced
            latest_func = torch.LongTensor([start_id] * bsz).to(device)  # [bsz, ]
            programs = [latest_func]
            last_h = pooler_output.unsqueeze(0)
            for i in range(self.max_program_len):
                p_word_emb = self.word_dropout(self.function_embeddings(latest_func)).unsqueeze(1)  # [bsz, 1, dim_w]
                p_word_h, last_h = self.function_decoder.forward_one_step(p_word_emb, last_h)  # [bsz, 1, dim_h]
                # attention over question words
                attn = torch.softmax(torch.bmm(p_word_h, sequence_output.permute(0, 2, 1)), dim=2)  # [bsz, 1, max_q]
                attn_word_h = torch.bmm(attn, sequence_output)  # [bsz, 1, dim_h]
                # sum up
                p_word_h = p_word_h + attn_word_h  # [bsz, 1, dim_h]

                # predict function
                logit_func = self.function_classifier(p_word_h).squeeze(1)  # [bsz, num_func]
                latest_func = torch.argmax(logit_func, dim=1)  # [bsz, ]
                programs.append(latest_func)
                finished = finished | latest_func.eq(end_id).byte()
                if finished.sum().item() == bsz:
                    # print('finished at step {}'.format(i))
                    break
            programs = torch.stack(programs, dim=1)  # [bsz, max_prog]
            outputs['pred_functions'] = programs

            # calculate attention of input sequence with programs
            function_ids = programs
            ## get embeddings of programs
            func_emb = self.word_dropout(self.function_embeddings(function_ids))  # [bsz, max_prog, dim_h]
            ## get length of specific sequences from start to end
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            ## get hidden states at specific time steps
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(),
                                                   h_0=pooler_output.unsqueeze(0))  # [bsz, max_prog, dim_h]
            ## calculate actual attention between hidden states and sequence words
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2)  # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output)  # [bsz, max_prog, dim_h]

            f_word_h = f_word_h + attn_word_h  # # [bsz, max_prog, dim_h]

            def pred_arguments(programs, f_word_h, input_embeddings, pred_type):

                ## get concept pos
                arguments_maps = []
                for program in programs:  # outputs['pred_functions']:

                    argument_pos = []
                    for idx, function in enumerate(program):

                        if pred_type == 'concept':
                            if function == self.vocab['function2id']['FilterConcept']:
                                argument_pos.append([idx])
                        elif pred_type == 'entity':
                            if function == self.vocab['function2id']['Find']:
                                argument_pos.append([idx])
                        elif pred_type == 'relation':
                            if function == self.vocab['function2id']['Relate']:
                                argument_pos.append([idx])
                        elif pred_type == 'attribute':
                            # if function == self.vocab['function2id']['QueryAttr']:#or :
                            if any([function == self.vocab['function2id'][qualifier] for qualifier in
                                    ['QueryAttr', 'FilterYear', 'FilterDate', 'FilterNum']]):
                                argument_pos.append([idx])
                        elif pred_type == 'operator':
                            if any([function == self.vocab['function2id'][qualifier] for qualifier in
                                    ['FilterYear', 'FilterDate', 'FilterNum']]):
                                argument_pos.append([idx])

                                # inputs = f['inputs']
                    #           #c = inputs[0]
                    #           #if not c in vocab['concept2id']:
                    #           #    continue
                    #           #c = vocab['concept2id'][c]

                    if len(argument_pos) == 0:
                        argument_pos.append([0])  # , vocab['entity2id']['<PAD>']])
                    arguments_maps.append(argument_pos)
                ## PADS sequence for max length of arguments in batch (Max two entities in one query in batch --> pad to length two
                max_args = max([len(example) for example in arguments_maps])
                # print(arguments_maps)
                for example in arguments_maps:
                    while len(example) < max_args:
                        example.append([0])

                argument_pos = torch.LongTensor(arguments_maps)
                dim_h = f_word_h.size(-1)

                argument_pos = torch.repeat_interleave(argument_pos, dim_h, dim=2)  ##[bsz, max_args, dim_h]
                arg_f_word_h = torch.gather(f_word_h, 1, argument_pos.to(device))  ## [bsz, max_args dim_h]

                # print("")
                # print("Printing tensors for hidden layer "+ f"{pred_type}")
                # print(arg_f_word_h.shape)
                # print(input_embeddings)

                # if pred_type == 'operator':

                if pred_type == 'concept':
                    class_embeddings = self.concept_classifier(input_embeddings)  ##[num_args, dim_h]
                elif pred_type == 'entity':
                    class_embeddings = self.entity_classifier(
                        input_embeddings)  # input_embeddings#model.entity_classifier(input_embeddings)
                elif pred_type == 'attribute':
                    class_embeddings = self.attribute_classifier(input_embeddings)
                    # class_embeddings = model.relation_classifier(input_embeddings)
                elif pred_type == 'relation':
                    class_embeddings = self.relation_classifier(input_embeddings)

                elif pred_type == 'operator':
                    operation_logits = self.operation_classifier(arg_f_word_h)
                    pred_concepts = [
                        [[argument_pos[0] for argument_pos in argument_map] for argument_map in arguments_maps],
                        torch.argmax(operation_logits, dim=2).cpu().numpy()]
                    return pred_concepts, arg_f_word_h, operation_logits

                    # print("")
                # print("Printing tensors for input embeddings "+ f"{pred_type}")
                # print(input_embeddings)

                # print("Printing tensors for class embeddings "+ f"{pred_type}")
                # #print("")
                # print(class_embeddings)
                argument_logits = arg_f_word_h @ class_embeddings.t()  # [bsz, num_concepts]
                # print("")
                # print("Printing tensors for argument logits "+ f"{pred_type}")
                # print(argument_logits)
                # print(argument_logits.shape)
                pred_concepts = [
                    [[argument_pos[0] for argument_pos in argument_map] for argument_map in arguments_maps],
                    torch.argmax(argument_logits, dim=2).cpu().numpy()]
                return pred_concepts, arg_f_word_h, argument_logits

            # pred_concepts = pred_arguments(programs, f_word_h, concept_embeddings, 'concept')
            outputs['pred_attributes'], hidden_att, att_embeds = pred_arguments(programs, f_word_h,
                                                                                attribute_embeddings,
                                                                                'attribute')
            outputs['pred_entities'], hidden_ent, ent_embeds = pred_arguments(programs, f_word_h,
                                                                              self.entity_embeddings,
                                                                              'entity')

            outputs['pred_relations'], _, _ = pred_arguments(programs, f_word_h, relation_embeddings, 'relation')
            outputs['pred_concepts'], _, _ = pred_arguments(programs, f_word_h, concept_embeddings, 'concept')

            outputs['pred_operators'], _, _ = pred_arguments(programs, f_word_h, "", 'operator')

            # outputs['pred_values'],_,_ = pred_arguments(programs, "","", 'values')

        ## Transforms raw predictions into KG entities
        ## Calculated Topk predicted indices of entities
        ## TopK
        topk = 5
        outputs['topk_entities'] = torch.topk(ent_embeds, topk, dim=2).indices.cpu().tolist()
        ## Transform indices into entities
        outputs['topk_entities'] = [
            [list(map(lambda entity: self.config.vocab['id2entity'][entity], group)) for group in example] for example
            in outputs['topk_entities']]
        ## Calculates scores and zips them together wiit predicted entities in first step ['Seymour Cassel',  'Sheldon Leonard',  'Mitch Pileggi',  'Gatineau',  '67th Academy Awards'], [0.85, 0.10, 0.02,0.01,0.01]
        transform1 = [list(zip(output[0], output[1])) for output in list(zip(outputs['topk_entities'], np.round(
            torch.topk(torch.softmax(ent_embeds, dim=2), topk, dim=2).values.cpu().numpy(), 4).astype(float)))]
        transform2 = [[list(zip(scores[0], scores[1])) for scores in output] for output in transform1]
        outputs['topk_entities'] = [outputs['pred_entities'][0], transform2]

        # outputs['pred_entities'][1] = [[self.config.vocab['id2entity'][pred[0]]] for pred in outputs['pred_entities'][1]
        # outputs['pred_concepts'][1] = [[self.config.vocab['id2concept'][pred[0]]] for pred in outputs['pred_concepts'][1]]
        # outputs['pred_attributes'][1] = [[self.config.vocab['id2attribute'][pred[0]]] for pred in outputs['pred_attributes'][1]]
        # outputs['pred_relations'][1] = [[self.config.vocab['id2relation'][pred[0]]] for pred in outputs['pred_relations'][1]]

        ## Transform indices into entities
        outputs['pred_entities'][1] = [list(map(lambda entity: self.config.vocab['id2entity'][entity], group)) for
                                       group in outputs['pred_entities'][1]]

        outputs['pred_concepts'][1] = [list(map(lambda entity: self.config.vocab['id2concept'][entity], group)) for
                                       group in outputs['pred_concepts'][1]]

        outputs['pred_attributes'][1] = [list(map(lambda entity: self.config.vocab['id2attribute'][entity], group)) for
                                         group in outputs['pred_attributes'][1]]

        outputs['pred_relations'][1] = [list(map(lambda entity: self.config.vocab['id2relation'][entity], group)) for
                                        group in outputs['pred_relations'][1]]
        ### Todo might need to be changed
        # outputs['pred_operators'][1] = [[self.config.vocab['id2operator'][pred[0]]] for pred in
        #                                outputs['pred_operators'][1]]

        outputs['pred_operators'][1] = [list(map(lambda entity: self.config.vocab['id2operator'][entity], group)) for
                                        group in outputs['pred_operators'][1]]
        # print(outputs['pred_functions'])
        # print(self.config.vocab['id2function'])
        ## Creates dictionary structure of prediction [{'function':"Find", 'inputs': []}, {'function':"Relate", 'inputs':[]}, ...]
        parsed = []
        predictions = [[self.config.vocab['id2function'][function.item()] for function in program] for program in
                       outputs['pred_functions'].cpu()]
        for prediction in predictions:
            # print(prediction[1:prediction.index('<END>')])
            # print(entities_pos)
            # print(entities_pred)
            temp = [{'function': function, 'inputs': []} for function in prediction[1:prediction.index('<END>')]]

            parsed.append(temp)
            # print({'function':function,'input':[]})

        ## Fills dictionary structure with right inputs  [{'function':"Find", 'inputs': ['COS B']}, {'function':"Relate", 'inputs':['LaunchVehicle' ]}, ...]
        ## Do not include following keys in prediction
        for key in list(outputs.keys() - {'pred_functions', 'pred_operators', 'topk_entities'}):
            for num_batch, parse in enumerate(parsed):

                ## for each argument in prediction
                for argument_num, pos in enumerate(outputs[key][0][num_batch]):
                    ## if argument is pad --> skip
                    if pos != 0:
                        ## add argument to right function
                        parse[pos - 1]['inputs'] = parse[pos - 1]['inputs'] + [outputs[key][1][num_batch][argument_num]]
        ## Insert operators last to retain position required for KG query
        key = 'pred_operators'
        # print(outputs[key])
        for num_batch, parse in enumerate(parsed):

            ## for each argument in prediction
            for argument_num, pos in enumerate(outputs[key][0][num_batch]):
                ## if argument is pad --> skip
                if pos != 0:
                    ## add argument to right function
                    parse[pos - 1]['inputs'] = parse[pos - 1]['inputs'] + [outputs[key][1][num_batch][argument_num]]
        return parsed, outputs

    ## adapted to attributes + values

    def forward(self, concept_inputs=None, relation_inputs=None, entity_inputs=None, attribute_inputs=None,
                input_ids=None, token_type_ids=None, attention_mask=None, function_ids=None, relation_info=None,
                concept_info=None, entity_info=None, attribute_info=None, operator_info=None):

        ## encode query with base BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # [bsz, max_seq_length, hidden_size]
        pooler_output = outputs[1]  # [bsz, hidden_size]
        outputs = {}
        sequence_output = self.dropout(sequence_output)

        # bsz = input_ids.size(0)

        def train(kb_inputs, kb_info, sequence_output, pooler_output, function_ids, pred_type):

            ##### PREDICT functions
            ## predicts the functions for the corresponding query for training
            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)

            ## predict hidden layers in GRU
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(),
                                                   h_0=pooler_output.unsqueeze(0))  # [bsz, max_prog, dim_h]

            ## calculate attention between hidden layers of GRU and token embeddings
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2)  # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output)  # [bsz, max_prog, dim_h]

            ## add attention to hidden layer output of GRU
            f_word_h = f_word_h + attn_word_h  # # [bsz, max_prog, dim_h]
            ## calculate logits of function predictor
            function_logits = self.function_classifier(f_word_h)
            outputs['function_logits'] = function_logits

            ## Change padding function to -100 to ignore in loss calculation
            mask_func_ids = copy.deepcopy(function_ids)
            mask_func_ids[mask_func_ids == 0] = -100
            # function_ids[function_ids == 0] = -100
            outputs['function_loss'] = nn.CrossEntropyLoss()(function_logits.permute(0, 2, 1)[:, :, :-1],
                                                             mask_func_ids[:, 1:])
            # PREDICT KB inputs
            ## sets the position of the to be predicted entries in the KB and their id in the vocab of the KB
            bsz = function_ids.size(0)
            kb_pos, kb_id = kb_info

            ## get "attentionised" hidden layer output of specific position in predictions
            dim_h = f_word_h.size(-1)
            kb_pos = kb_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, kb_pos).squeeze(1)  # [bsz, dim_h]

            ## if operation, use hidden layer for classification
            if pred_type == 'operation':
                operation_logits = self.operation_classifier(f_word_h)
                outputs['operator_logits'] = operation_logits
                # print('operator_logits', operation_logits.size())
                # print('operator_id', kb_id.size())
                kb_id = kb_id.squeeze(-1)
                outputs['operator_loss'] = nn.CrossEntropyLoss()(operation_logits, kb_id)
                return outputs

            ## else embed the to be linked entities / relations / attributes / etc.
            embeddings_kb = self.bert(input_ids=kb_inputs['input_ids'], \
                                      attention_mask=kb_inputs['attention_mask'], \
                                      token_type_ids=kb_inputs['token_type_ids'])[1]  # [num_relations, dim_h]

            ## chose type of prediction
            if pred_type == 'entity':
                ## transform embeddings in classifier layer
                embeddings_kb = self.entity_classifier(embeddings_kb)  # [num_relations, dim_h]
                ## calculate dot product with "attentionised" hidden layer
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['entity_logits'] = kb_logits
                kb_id = kb_id.squeeze(-1)
                # print('entity_logits', kb_logits.size())
                # print('entity_id', kb_id.size())
                outputs['entity_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                return outputs

            elif pred_type == 'concept':
                embeddings_kb = self.concept_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['concept_logits'] = kb_logits
                kb_id = kb_id.squeeze(-1)
                # print('concept_logits', kb_logits.size())
                # print('concept_id', kb_id.size())
                outputs['concept_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                return outputs

            elif pred_type == 'relation':
                embeddings_kb = self.relation_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['relation_logits'] = kb_logits
                kb_id = kb_id.squeeze(-1)
                # print('relation_logits',  kb_logits.size())
                # print('relation_id', kb_id.size())
                outputs['relation_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                return outputs

            elif pred_type == 'attribute':
                embeddings_kb = self.attribute_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['attribute_logits'] = kb_logits
                kb_id = kb_id.squeeze(-1)
                # print('relation_logits',  kb_logits.size())
                # print('relation_id', kb_id.size())
                outputs['attribute_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                return outputs

        def evaluate(kb_inputs, kb_info, sequence_output, pooler_output, function_ids, pred_type):

            # PREDICT functions
            ## predicts the functions for the corresponding query for evaluation
            bsz = function_ids.size(0)
            device = function_ids.device
            start_id = self.vocab['function2id']['<START>']
            end_id = self.vocab['function2id']['<END>']
            finished = torch.zeros((bsz,)).byte().to(device)  # record whether <END> is produced
            latest_func = torch.LongTensor([start_id] * bsz).to(device)  # [bsz, ]
            programs = [latest_func]

            last_h = pooler_output.unsqueeze(0)
            function_logits = []
            for i in range(self.max_program_len):
                p_word_emb = self.word_dropout(self.function_embeddings(latest_func)).unsqueeze(1)  # [bsz, 1, dim_w]
                p_word_h, last_h = self.function_decoder.forward_one_step(p_word_emb, last_h)  # [bsz, 1, dim_h]
                # attention over question words
                attn = torch.softmax(torch.bmm(p_word_h, sequence_output.permute(0, 2, 1)), dim=2)  # [bsz, 1, max_q]
                attn_word_h = torch.bmm(attn, sequence_output)  # [bsz, 1, dim_h]
                # sum up
                p_word_h = p_word_h + attn_word_h  # [bsz, 1, dim_h]

                # predict function
                logit_func = self.function_classifier(p_word_h).squeeze(1)  # [bsz, num_func]
                function_logits.append(logit_func)

                latest_func = torch.argmax(logit_func, dim=1)  # [bsz, ]
                programs.append(latest_func)
                finished = finished | latest_func.eq(end_id).byte()
                if finished.sum().item() == bsz:
                    # print('finished at step {}'.format(i))
                    break
            programs = torch.stack(programs, dim=1)  # [bsz, max_prog]
            outputs['pred_functions'] = programs
            # mask_func_ids = copy.deepcopy(function_ids)
            # mask_func_ids[mask_func_ids == 0] = -100
            function_ids[function_ids == 0] = -100
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)

            ## Transforms list of logit tensor for each time step to tensor
            function_logits = torch.stack(function_logits, dim=1)

            ## eventually pads tensor depending on if prediction of max_length is correct
            current_shape = function_logits.shape
            pad_top = max(0, max(func_lens) - current_shape[1])  #
            padding = (0, 0, 0, pad_top)
            padded_tensor = torch.nn.functional.pad(function_logits, padding, "constant", 0)

            ## eventually truncates predictions down to max length of functions -1
            padded_tensor = padded_tensor.narrow(1, 0, max(func_lens - 1))

            ## Cuts function_ids down to max programn length
            function_ids = function_ids.narrow(1, 0, max(func_lens))

            ## Calculates validation loss from predicted programs (padded_tensor) and
            ## ground truth (function_ids), which are indexed after the starting program
            # loss = torch.nn.CrossEntropyLoss()(padded_tensor.permute(0, 2, 1),
            #                                   mask_func_ids[:, 1:])
            loss = torch.nn.CrossEntropyLoss()(padded_tensor.permute(0, 2, 1),
                                               function_ids[:, 1:])
            outputs['function_loss'] = loss

            function_ids[function_ids == -100] = 0
            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            # func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(),
                                                   h_0=pooler_output.unsqueeze(0))  # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2)  # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output)  # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h  # # [bsz, max_prog, dim_h]
            # relation_pos = [relation_pos] * self.hidden_size
            # a : [bsz, max_prog, dim_h]
            # b : [bsz, 1]
            # c = b.repeat(1, dim_h).view(bsz,1,dim_h)
            # a.gather(1,c).view((bsz, dim_h))

            dim_h = f_word_h.size(-1)
            kb_pos, kb_id = kb_info
            # print(kb_pos)
            kb_pos = kb_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, kb_pos).squeeze(1)  # [bsz, dim_h]

            if pred_type == 'operation':
                operation_logits = self.operation_classifier(f_word_h)
                outputs['operator_logits'] = operation_logits
                outputs['pred_operator'] = torch.argmax(operation_logits, dim=1)
                # outputs['operator_loss'] = nn.CrossEntropyLoss()(operation_logits, kb_id)
                return outputs

            if pred_type == 'entity':
                embeddings_kb = self.entity_embeddings

            else:
                embeddings_kb = self.bert(input_ids=kb_inputs['input_ids'], \
                                          attention_mask=kb_inputs['attention_mask'], \
                                          token_type_ids=kb_inputs['token_type_ids'])[1]  # [num_relations, dim_h]

            if pred_type == 'entity':
                embeddings_kb = self.entity_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['entity_logits'] = kb_logits
                #
                # print('entity_logits', kb_logits.size())
                # print('entity_id', kb_id.size())

                outputs['pred_entity'] = torch.argmax(kb_logits, dim=1)
                # kb_id = kb_id.squeeze(-1)
                # outputs['entity_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                return outputs

            elif pred_type == 'concept':
                embeddings_kb = self.concept_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['concept_logits'] = kb_logits
                # kb_id = kb_id.squeeze(-1)
                # print('concept_logits', kb_logits.size())
                # print('concept_id', kb_id.size())
                # outputs['concept_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                outputs['pred_concept'] = torch.argmax(kb_logits, dim=1)
                return outputs

            elif pred_type == 'relation':
                embeddings_kb = self.relation_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['relation_logits'] = kb_logits
                # kb_id = kb_id.squeeze(-1)
                # print('relation_logits',  kb_logits.size())
                # print('relation_id', kb_id.size())
                # outputs['relation_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                outputs['pred_relation'] = torch.argmax(kb_logits, dim=1)
                return outputs

            elif pred_type == 'attribute':
                embeddings_kb = self.attribute_classifier(embeddings_kb)  # [num_relations, dim_h]
                kb_logits = f_word_h @ embeddings_kb.t()  # [bsz, num_relationis]
                outputs['attribute_logits'] = kb_logits
                # kb_id = kb_id.squeeze(-1)
                # print('relation_logits',  kb_logits.size())
                # print('relation_id', kb_id.size())
                # outputs['attribute_loss'] = nn.CrossEntropyLoss()(kb_logits, kb_id)
                outputs['pred_attribute'] = torch.argmax(kb_logits, dim=1)
                return outputs

        ## Train
        if entity_info is not None and entity_info[1] is not None:
            outputs = train(entity_inputs, entity_info, sequence_output, pooler_output, function_ids, 'entity')

        if relation_info is not None and relation_info[1] is not None:
            outputs = train(relation_inputs, relation_info, sequence_output, pooler_output, function_ids, 'relation')

        if concept_info is not None and concept_info[1] is not None:
            outputs = train(concept_inputs, concept_info, sequence_output, pooler_output, function_ids, 'concept')

        if attribute_info is not None and attribute_info[1] is not None:
            outputs = train(attribute_inputs, attribute_info, sequence_output, pooler_output, function_ids, 'attribute')

        if operator_info is not None and operator_info[1] is not None:
            outputs = train('', operator_info, sequence_output, pooler_output, function_ids, 'operation')

        ## Eval
        if entity_info is not None and entity_info[1] is None:
            outputs = evaluate(entity_inputs, entity_info, sequence_output, pooler_output, function_ids, 'entity')

        if relation_info is not None and relation_info[1] is None:
            outputs = evaluate(relation_inputs, relation_info, sequence_output, pooler_output, function_ids, 'relation')

        if concept_info is not None and concept_info[1] is None:
            outputs = evaluate(concept_inputs, concept_info, sequence_output, pooler_output, function_ids, 'concept')

        if attribute_info is not None and attribute_info[1] is None:
            outputs = evaluate(attribute_inputs, attribute_info, sequence_output, pooler_output, function_ids,
                               'attribute')

        if operator_info is not None and operator_info[1] is None:
            outputs = evaluate('', operator_info, sequence_output, pooler_output, function_ids, 'operation')

        return outputs