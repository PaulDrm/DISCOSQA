import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import *
from transformers import (BertConfig, BertModel,  BertPreTrainedModel)
from Pretraining.utils import *
from Pretraining.BiGRU import GRU, BiGRU
import pickle

class RelationPT(BertPreTrainedModel):
    def __init__(self, config):
        super(RelationPT, self).__init__(config)
        self.vocab = config.vocab
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_functions = len(config.vocab['function2id'])
        self.function_embeddings = nn.Embedding(self.num_functions, config.hidden_size)
        self.function_decoder = GRU(config.hidden_size, config.hidden_size, num_layers = 1, dropout = 0.2)
        self.function_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_functions),
        )
        self.word_dropout = nn.Dropout(0.2)
        self.max_program_len = 17
        # self.relation_inputs = config.relation_inputs
        self.relation_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )

        self.concept_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )
        # ## Todo classifier changed

        self.entity_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.hidden_size),
        )

        # ## Todo classifier changed
        # self.attribute_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, config.hidden_size),
        # )
        # self.values_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, config.hidden_size),
        # )

        self.hidden_size = config.hidden_size
        self.init_weights()
    
    def demo(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0] # [bsz, max_seq_length, hidden_size]
        pooler_output = outputs[1] # [bsz, hidden_size]
        outputs = {}
        sequence_output = self.dropout(sequence_output)
        bsz = input_ids.size(0)
        device = input_ids.device
        start_id = self.vocab['function2id']['<START>']
        end_id = self.vocab['function2id']['<END>']
        finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced
        latest_func = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
        programs = [latest_func]
        last_h = pooler_output.unsqueeze(0)
        for i in range(self.max_program_len):
            p_word_emb = self.word_dropout(self.function_embeddings(latest_func)).unsqueeze(1) # [bsz, 1, dim_w]
            p_word_h, last_h = self.function_decoder.forward_one_step(p_word_emb, last_h) # [bsz, 1, dim_h]
            # attention over question words
            attn = torch.softmax(torch.bmm(p_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, 1, max_q]
            attn_word_h = torch.bmm(attn, sequence_output) # [bsz, 1, dim_h]
            # sum up
            p_word_h = p_word_h + attn_word_h # [bsz, 1, dim_h]

            # predict function
            logit_func = self.function_classifier(p_word_h).squeeze(1) # [bsz, num_func]
            latest_func = torch.argmax(logit_func, dim=1) # [bsz, ]
            programs.append(latest_func)
            finished = finished | latest_func.eq(end_id).byte()
            if finished.sum().item() == bsz:
                # print('finished at step {}'.format(i))
                break
        programs = torch.stack(programs, dim=1) # [bsz, max_prog]
        outputs['pred_functions'] = programs
        return outputs

    ## adapted to attributes + values

    def forward(self, concept_inputs, relation_inputs, entity_inputs, input_ids, token_type_ids, attention_mask, function_ids, relation_info, concept_info, entity_info, entity_embeddings):#, attribute_info, value_info):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0] # [bsz, max_seq_length, hidden_size]
        pooler_output = outputs[1] # [bsz, hidden_size]
        outputs = {}
        sequence_output = self.dropout(sequence_output)
        bsz = input_ids.size(0)

        #def function_pred:
        ## Todo changed: inserted entity info
        if entity_info is not None and entity_info[1] is not None:
            concept_info = entity_info
            concept_pos, concept_id = concept_info
            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(),
                                                   h_0=pooler_output.unsqueeze(0))  # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2)  # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output)  # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h  # # [bsz, max_prog, dim_h]
            function_logits = self.function_classifier(f_word_h)
            outputs['function_logits'] = function_logits
            outputs['function_loss'] = nn.CrossEntropyLoss()(function_logits.permute(0, 2, 1)[:, :, :-1],
                                                             function_ids[:, 1:])
            # outputs['relation_loss'] = outputs['function_loss']
            dim_h = f_word_h.size(-1)
            concept_pos = concept_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, concept_pos).squeeze(1)  # [bsz, dim_h]
            # concept_embeddings = self.bert(input_ids=concept_inputs['input_ids'], \
            #                                attention_mask=concept_inputs['attention_mask'], \
            #                                token_type_ids=concept_inputs['token_type_ids'])[1]  # [num_relations, dim_h]
            #with torch.no_grad():
            #print('memory')
            #concept_embeddings = self.bert(input_ids=entity_inputs['input_ids'], \
            #                                   attention_mask = entity_inputs['attention_mask'], \
            #                                   token_type_ids = entity_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            #with open('c_embeddings.pt', 'rb') as f:
            #    # for _ in range(3):
            #    concept_embeddings = pickle.load(f)

            #with torch.no_grad():
            #    concept_embeddings = self.bert(input_ids=entity_inputs['input_ids'],
            #                                    attention_mask=entity_inputs['attention_mask'],
            #                                    token_type_ids=entity_inputs['token_type_ids'])[1]
            #print(concept_embeddings)
            entity_embeddings = self.bert(input_ids=entity_inputs['input_ids'], \
                                           attention_mask = entity_inputs['attention_mask'], \
                                           token_type_ids = entity_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            concept_embeddings = self.entity_classifier(entity_embeddings)  # [num_relations, dim_h]
            #print('concept_logits',concept_embeddings.size())
            concept_logits = f_word_h @ concept_embeddings.t()  # [bsz, num_relations]

            ## Load external saved embeddings for vocab
            #with open('c_embeddings.pt', 'rb') as f:
            #    # for _ in range(3):
            #    concept_embeddings = pickle.load(f)
            #concept_embeddings = self.entity_classifier(concept_embeddings)  # [num_concepts, dim_h]

            #concept_logits = f_word_h @ concept_embeddings.t()  # [bsz, num_relations]

            outputs['entity_logits'] = concept_logits
            #concept_id = concept_id.squeeze(-1)
            # print('relation_logits', relation_logits.size())
            # print('relation_id', relation_id.size())
            #print('concept_logits',concept_logits.size())
            #print('concept_logits',concept_logits.size())
            #print(concept_logits.max(), concept_logits.min())
            #torch.set_printoptions(profile="full")
            #print(concept_logits)
            #torch.set_printoptions(profile="default")
            #concept_id = torch.LongTensor([1]).cuda()
            #concept_id = [i for i, _ in enumerate(set(concept_id))]
            #print(concept_id)
            #print(concept_logits.max(), concept_logits.min())
            #concept_id = torch.LongTensor(concept_id).cuda()
            #print('concept_ids', concept_id.size())
            #print(concept_id)
            #outputs['entity_loss'] = F.cross_entropy(concept_logits, concept_id)

            #print('concept_ids', concept_id.size())

            outputs['entity_loss'] = nn.CrossEntropyLoss()(concept_logits, concept_id)
            #print(outputs)
            concept_info = None


            # ## predicting functions
            # func_emb = self.word_dropout(self.function_embeddings(function_ids))
            # func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            # f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(), h_0=pooler_output.unsqueeze(0)) # [bsz, max_prog, dim_h]
            # attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
            # attn_word_h = torch.bmm(attn, sequence_output) # [bsz, max_prog, dim_h]
            # f_word_h = f_word_h + attn_word_h # # [bsz, max_prog, dim_h]
            # function_logits = self.function_classifier(f_word_h)
            # outputs['function_logits'] = function_logits
            # outputs['function_loss'] = nn.CrossEntropyLoss()(function_logits.permute(0, 2, 1)[:,:,:-1], function_ids[:,1:])
            # # outputs['relation_loss'] = outputs['function_loss']
            # ## predicting entities
            # entity_pos, entity_id = entity_info
            # dim_h = f_word_h.size(-1)
            # entity_pos = entity_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            # f_word_h = torch.gather(f_word_h, 1, entity_pos).squeeze(1) # [bsz, dim_h]
            # entity_embeddings = self.bert(input_ids = entity_inputs['input_ids'], \
            # attention_mask = entity_inputs['attention_mask'], \
            # token_type_ids = entity_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            # entity_embeddings = self.entity_classifier(entity_embeddings) # [num_relations, dim_h]
            # entity_logits = f_word_h @ entity_embeddings.t() # [bsz, num_relationis]
            # outputs['entity_logits'] = entity_logits
            # entity_id = entity_id.squeeze(-1)
            # # print('relation_logits', relation_logits.size())
            # # print('relation_id', relation_id.size())
            # outputs['entity_loss'] = nn.CrossEntropyLoss()(entity_logits, entity_id)



        if relation_info is not None and relation_info[1] is not None:
            relation_pos, relation_id = relation_info
            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(), h_0=pooler_output.unsqueeze(0)) # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output) # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h # # [bsz, max_prog, dim_h]
            function_logits = self.function_classifier(f_word_h)
            outputs['function_logits'] = function_logits
            outputs['function_loss'] = nn.CrossEntropyLoss()(function_logits.permute(0, 2, 1)[:,:,:-1], function_ids[:,1:])
            # outputs['relation_loss'] = outputs['function_loss']
            dim_h = f_word_h.size(-1)
            relation_pos = relation_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, relation_pos).squeeze(1) # [bsz, dim_h]
            relation_embeddings = self.bert(input_ids = relation_inputs['input_ids'], \
            attention_mask = relation_inputs['attention_mask'], \
            token_type_ids = relation_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            relation_embeddings = self.relation_classifier(relation_embeddings) # [num_relations, dim_h]
            relation_logits = f_word_h @ relation_embeddings.t() # [bsz, num_relationis]
            outputs['relation_logits'] = relation_logits
            relation_id = relation_id.squeeze(-1)
            # print('relation_logits', relation_logits.size())
            # print('relation_id', relation_id.size())
            outputs['relation_loss'] = nn.CrossEntropyLoss()(relation_logits, relation_id)

        if concept_info is not None and concept_info[1] is not None:
            concept_pos, concept_id = concept_info
            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(), h_0=pooler_output.unsqueeze(0)) # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output) # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h # # [bsz, max_prog, dim_h]
            function_logits = self.function_classifier(f_word_h)
            outputs['function_logits'] = function_logits
            outputs['function_loss'] = nn.CrossEntropyLoss()(function_logits.permute(0, 2, 1)[:,:,:-1], function_ids[:,1:])
            # outputs['relation_loss'] = outputs['function_loss']
            dim_h = f_word_h.size(-1)
            concept_pos = concept_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, concept_pos).squeeze(1) # [bsz, dim_h]
            concept_embeddings = self.bert(input_ids = concept_inputs['input_ids'], \
            attention_mask = concept_inputs['attention_mask'], \
            token_type_ids = concept_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            concept_embeddings = self.concept_classifier(concept_embeddings) # [num_relations, dim_h]
            concept_logits = f_word_h @ concept_embeddings.t() # [bsz, num_relations]
            outputs['concept_logits'] = concept_logits
            concept_id = concept_id.squeeze(-1)
            # print('relation_logits', relation_logits.size())
            # print('relation_id', relation_id.size())
            outputs['concept_loss'] = nn.CrossEntropyLoss()(concept_logits, concept_id)
        

        if entity_info is not None and entity_info[1] is None:
            relation_pos, relation_id = entity_info
            bsz = input_ids.size(0)
            device = input_ids.device
            start_id = self.vocab['function2id']['<START>']
            end_id = self.vocab['function2id']['<END>']
            finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced
            latest_func = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
            programs = [latest_func]
            last_h = pooler_output.unsqueeze(0)
            for i in range(self.max_program_len):
                p_word_emb = self.word_dropout(self.function_embeddings(latest_func)).unsqueeze(1) # [bsz, 1, dim_w]
                p_word_h, last_h = self.function_decoder.forward_one_step(p_word_emb, last_h) # [bsz, 1, dim_h]
                # attention over question words
                attn = torch.softmax(torch.bmm(p_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, 1, max_q]
                attn_word_h = torch.bmm(attn, sequence_output) # [bsz, 1, dim_h]
                # sum up
                p_word_h = p_word_h + attn_word_h # [bsz, 1, dim_h]

                # predict function
                logit_func = self.function_classifier(p_word_h).squeeze(1) # [bsz, num_func]
                latest_func = torch.argmax(logit_func, dim=1) # [bsz, ]
                programs.append(latest_func)
                finished = finished | latest_func.eq(end_id).byte()
                if finished.sum().item() == bsz:
                    # print('finished at step {}'.format(i))
                    break
            programs = torch.stack(programs, dim=1) # [bsz, max_prog]
            outputs['pred_functions'] = programs

            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(), h_0=pooler_output.unsqueeze(0)) # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output) # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h # # [bsz, max_prog, dim_h]
            # relation_pos = [relation_pos] * self.hidden_size
            # a : [bsz, max_prog, dim_h]
            # b : [bsz, 1]
            # c = b.repeat(1, dim_h).view(bsz,1,dim_h)
            # a.gather(1,c).view((bsz, dim_h))
            dim_h = f_word_h.size(-1)
            relation_pos = relation_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, relation_pos).squeeze(1) # [bsz, dim_h]
            #relation_embeddings = self.bert(input_ids = entity_inputs['input_ids'], \
            #attention_mask = relation_inputs['attention_mask'], \
            #token_type_ids = relation_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            #with open('c_embeddings.pt', 'rb') as f:
            ##    # for _ in range(3):
            #    relation_embeddings = pickle.load(f)
            relation_embeddings = entity_embeddings
            relation_embeddings = self.relation_classifier(relation_embeddings) # [num_relations, dim_h]
            relation_logits = f_word_h @ relation_embeddings.t() # [bsz, num_relationis]
            outputs['pred_entity'] = torch.argmax(relation_logits, dim = 1)


        if relation_info is not None and relation_info[1] is None:
            relation_pos, relation_id = relation_info
            bsz = input_ids.size(0)
            device = input_ids.device
            start_id = self.vocab['function2id']['<START>']
            end_id = self.vocab['function2id']['<END>']
            finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced
            latest_func = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
            programs = [latest_func]
            logits = []
            last_h = pooler_output.unsqueeze(0)
            for i in range(self.max_program_len):
                p_word_emb = self.word_dropout(self.function_embeddings(latest_func)).unsqueeze(1) # [bsz, 1, dim_w]
                p_word_h, last_h = self.function_decoder.forward_one_step(p_word_emb, last_h) # [bsz, 1, dim_h]
                # attention over question words
                attn = torch.softmax(torch.bmm(p_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, 1, max_q]
                attn_word_h = torch.bmm(attn, sequence_output) # [bsz, 1, dim_h]
                # sum up
                p_word_h = p_word_h + attn_word_h # [bsz, 1, dim_h]

                # predict function
                logit_func = self.function_classifier(p_word_h).squeeze(1) # [bsz, num_func]

                latest_func = torch.argmax(logit_func, dim=1) # [bsz, ]
                programs.append(latest_func)
                finished = finished | latest_func.eq(end_id).byte()
                if finished.sum().item() == bsz:
                    # print('finished at step {}'.format(i))
                    break
            programs = torch.stack(programs, dim=1) # [bsz, max_prog]
            outputs['pred_functions'] = programs

            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(), h_0=pooler_output.unsqueeze(0)) # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output) # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h # # [bsz, max_prog, dim_h]
            # relation_pos = [relation_pos] * self.hidden_size
            # a : [bsz, max_prog, dim_h]
            # b : [bsz, 1]
            # c = b.repeat(1, dim_h).view(bsz,1,dim_h)
            # a.gather(1,c).view((bsz, dim_h))
            dim_h = f_word_h.size(-1)
            relation_pos = relation_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, relation_pos).squeeze(1) # [bsz, dim_h]
            relation_embeddings = self.bert(input_ids = relation_inputs['input_ids'], \
            attention_mask = relation_inputs['attention_mask'], \
            token_type_ids = relation_inputs['token_type_ids'])[1] # [num_relations, dim_h]
            relation_embeddings = self.relation_classifier(relation_embeddings) # [num_relations, dim_h]
            relation_logits = f_word_h @ relation_embeddings.t() # [bsz, num_relationis]
            outputs['pred_relation'] = torch.argmax(relation_logits, dim = 1)


        if concept_info is not None and concept_info[1] is None:
            concept_pos, concept_id = concept_info
            bsz = input_ids.size(0)
            device = input_ids.device
            start_id = self.vocab['function2id']['<START>']
            end_id = self.vocab['function2id']['<END>']
            finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced
            latest_func = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
            programs = [latest_func]
            last_h = pooler_output.unsqueeze(0)
            for i in range(self.max_program_len):
                p_word_emb = self.word_dropout(self.function_embeddings(latest_func)).unsqueeze(1) # [bsz, 1, dim_w]
                p_word_h, last_h = self.function_decoder.forward_one_step(p_word_emb, last_h) # [bsz, 1, dim_h]
                # attention over question words
                attn = torch.softmax(torch.bmm(p_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, 1, max_q]
                attn_word_h = torch.bmm(attn, sequence_output) # [bsz, 1, dim_h]
                # sum up
                p_word_h = p_word_h + attn_word_h # [bsz, 1, dim_h]

                # predict function
                logit_func = self.function_classifier(p_word_h).squeeze(1) # [bsz, num_func]
                latest_func = torch.argmax(logit_func, dim=1) # [bsz, ]
                programs.append(latest_func)
                finished = finished | latest_func.eq(end_id).byte()
                if finished.sum().item() == bsz:
                    # print('finished at step {}'.format(i))
                    break
            programs = torch.stack(programs, dim=1) # [bsz, max_prog]
            outputs['pred_functions'] = programs

            func_emb = self.word_dropout(self.function_embeddings(function_ids))
            func_lens = function_ids.size(1) - function_ids.eq(0).long().sum(dim=1)
            f_word_h, _, _ = self.function_decoder(func_emb, func_lens.cpu(), h_0=pooler_output.unsqueeze(0)) # [bsz, max_prog, dim_h]
            attn = torch.softmax(torch.bmm(f_word_h, sequence_output.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
            attn_word_h = torch.bmm(attn, sequence_output) # [bsz, max_prog, dim_h]
            f_word_h = f_word_h + attn_word_h # # [bsz, max_prog, dim_h]
            # concept_pos = [concept_pos] * self.hidden_size
            # a : [bsz, max_prog, dim_h]
            # b : [bsz, 1]
            # c = b.repeat(1, dim_h).view(bsz,1,dim_h)
            # a.gather(1,c).view((bsz, dim_h))
            dim_h = f_word_h.size(-1)
            concept_pos = concept_pos.repeat(1, dim_h).view(bsz, 1, dim_h)
            f_word_h = torch.gather(f_word_h, 1, concept_pos).squeeze(1) # [bsz, dim_h]
            concept_embeddings = self.bert(input_ids = concept_inputs['input_ids'], \
            attention_mask = concept_inputs['attention_mask'], \
            token_type_ids = concept_inputs['token_type_ids'])[1] # [num_concepts, dim_h]
            concept_embeddings = self.concept_classifier(concept_embeddings) # [num_concepts, dim_h]
            concept_logits = f_word_h @ concept_embeddings.t() # [bsz, num_conceptis]
            outputs['pred_concept'] = torch.argmax(concept_logits, dim = 1)

        return outputs
