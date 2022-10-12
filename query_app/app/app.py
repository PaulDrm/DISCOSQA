from KoPL_main.src.kopl.kopl import KoPLEngine
from Pretraining.utils import *
from Pretraining.model import RelationPT
from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)
import json
import copy
import time
import streamlit as st
import pickle
import os


@st.experimental_singleton
def load_kg(path):
    print("Load KG..")
    with open(path, 'r') as f:
        kqa_kb = json.load(f)
    kb = copy.deepcopy(kqa_kb)
    engine = KoPLEngine(kb)
    return engine

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2function'] = invert_dict(vocab['function2id'])
    return vocab

def parse_program(program):
    string = ""
    for function in reversed(program):
        if function.get('dependencies') != []:
            addition = parse_program(program[:-1])
            # print(function['function'])
            if function['inputs'] != []:
                string += addition + ','
                string = "engine." + function['function'] + '(' + string + ", ".join(
                    [f"'{inp}'" for inp in function['inputs']]) + ")"
            else:
                string += addition
                string = "engine." + function['function'] + '(' + string + ", ".join(
                    [f"'{inp}'" for inp in function['inputs']]) + ")"
            return string
        else:
            # print(program)
            # print(function['function'])
            # string += "engine."+function['function'] + "("+"', '".join(function['inputs'])+")"#+','
            # string += "engine."+function['function'] + "("+"', '".join(function['inputs'])+")"#+','
            string += "engine." + function['function'] + '(' + string + ", ".join(
                [f"'{inp}'" for inp in function['inputs']]) + ")"
            return string


def load_classes(path,device):
  with open(os.path.abspath(path), 'rb') as f:
      input_ids = pickle.load(f)
      token_type_ids = pickle.load(f)
      attention_mask = pickle.load(f)
      # input_ids = torch.LongTensor(input_ids[:512,:]).to(device)
      # token_type_ids = torch.LongTensor(token_type_ids[:512,:]).to(device)
      # attention_mask = torch.LongTensor(attention_mask[:512,:]).to(device)
      #input_ids = torch.LongTensor(input_ids).to(device)
      #token_type_ids = torch.LongTensor(token_type_ids).to(device)
      #attention_mask = torch.LongTensor(attention_mask).to(device)
      input_ids = torch.tensor(input_ids).to(device)
      token_type_ids = torch.tensor(token_type_ids).to(device)
      attention_mask = torch.tensor(attention_mask).to(device)
  argument_inputs = {
    'input_ids': input_ids,
    'token_type_ids': token_type_ids,
    'attention_mask': attention_mask
  }
  return argument_inputs
@st.experimental_singleton
def load_model():

    """
    Load the model from checkpoint
    """

    save_dir = "PaulD/checkpoint-14399"
    config_class, model_class = (BertConfig, RelationPT)
    print("load ckpt from {}".format(save_dir))
    config = config_class.from_pretrained(save_dir)  # , num_labels = len(label_list))
    model = model_class.from_pretrained(save_dir, config=config)
    path = './processed/vocab.json'
    model.config.vocab = load_vocab(path)

    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():  #
        model.cuda()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
    return model

@st.experimental_singleton
def load_tokenizer(path):
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(path, do_lower_case=False)
    return tokenizer

DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "What is the mass of COS B?")

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def main():
    ################################################################
    ## Load KG engine
    # path = 'C:\\Users\\pauld\\Projects\\IOA\\1_4_SW\\esa_kb.json'
    path = './Preprocessing_KG/esa_kb.json' # 'C:\\Users\\pauld\\Projects\\IOA\\3_1_Demo\\Preprocessing_KG\\esa_kb.json'
    engine = load_kg(os.path.abspath(path))

    # save_dir = '/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-14399'
    tokenizer =load_tokenizer('bert-base-cased')

    path = './processed/vocab.json'
    vocab = load_vocab(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = './processed/'
    argument_inputs = load_classes(input_dir + "entity/entity_small.pt", device)

    ## needs replacement
    ents = tokenizer.batch_decode(argument_inputs['input_ids'], skip_special_tokens=True)
    vocab['entity2id'] = {}
    for entity in ents:
        if not entity in vocab['entity2id']:
            vocab['entity2id'][entity] = len(vocab['entity2id'])
    vocab['id2entity'] = [entity for entity, iid in vocab['entity2id'].items()]
    model = load_model()
    model.config.vocab = vocab

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)

    # argument_inputs = load_classes(input_dir + "entity/entity.pt", device)
    # with torch.no_grad():
    #     model.entity_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
    #                                    attention_mask=argument_inputs['attention_mask'],
    #                                    token_type_ids=argument_inputs['token_type_ids'])[1]

    ## Loads embeddings
    with open(os.path.abspath(input_dir + "entity/entity_embeddings.pt"), 'rb') as f:
        model.entity_embeddings = pickle.load(f)

    print(st.session_state.get('attribute_embeddings')==None)

    if st.session_state.get('attribute_embeddings') == None:
        argument_inputs = load_classes(input_dir + "attribute/attribute.pt", device)
        with torch.no_grad():
            attribute_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
                                              attention_mask=argument_inputs['attention_mask'],
                                              token_type_ids=argument_inputs['token_type_ids'])[1]
            set_state_if_absent('attribute_embeddings', attribute_embeddings)

    if st.session_state.get('concept_embeddings') == None:
        argument_inputs = load_classes(input_dir + "concept/concept.pt", device)
        with torch.no_grad():
            concept_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
                                            attention_mask=argument_inputs['attention_mask'],
                                            token_type_ids=argument_inputs['token_type_ids'])[1]
            set_state_if_absent('concept_embeddings', concept_embeddings)

    argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
    with torch.no_grad():
        relation_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
                                         attention_mask=argument_inputs['attention_mask'],
                                         token_type_ids=argument_inputs['token_type_ids'])[1]

    if st.session_state.get('relation_embeddings') == None:
        argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
        with torch.no_grad():
            relation_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
                                             attention_mask=argument_inputs['attention_mask'],
                                             token_type_ids=argument_inputs['token_type_ids'])[1]
        set_state_if_absent('relation_embeddings', relation_embeddings)

    st.title("Demo for querying KG with ProgramTransfer method")
    st.markdown(""" *Note: do not use keywords, but full-fledged questions.*""")
    keyword = st.text_input('Enter a query for the KG','What is the mass of COS B?')
                            #"engine.QueryAttr(engine.Find('GAIA'),'Mass')")

    inputs = tokenizer([keyword], return_tensors='pt', padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs.keys()}

    #Enter a prediction of a model for querying the KB


    #object = model.demo(**inputs,relation_embeddings=relation_embeddings,concept_embeddings=concept_embeddings,attribute_embeddings=attribute_embeddings )[0]
    object = model.demo(**inputs, relation_embeddings=st.session_state.relation_embeddings, concept_embeddings=st.session_state.concept_embeddings,
                        attribute_embeddings=st.session_state.attribute_embeddings)[0]


    for count, function in enumerate(object):
        if count == 0:
            function.update({'dependencies': []})
        else:
            function.update({'dependencies': [-1 + count]})

    st.write(f"Predicted program: {str(object)}")



    #object = eval(keyword)
    #object
    
    try:
        result = eval(parse_program(object))
        if len(result)==1:
            st.write(result[0].value)

        is_correct_answer = None
        is_correct_document = None
        button_col1, button_col2, _ = st.columns([1, 1, 6])

        if button_col1.button("👍", help="Correct answer"):  # key=f"{result['context']}{count}1",
            is_correct_answer = True
            is_correct_document = True

        if button_col2.button("👎", help="Wrong answer"):  # key=f"{result['context']}{count}2",
            is_correct_answer = False
            is_correct_document = False
    except:
        #st.write
        st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")

main()