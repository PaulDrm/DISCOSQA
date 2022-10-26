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
#from streamlit_agraph import agraph, Node, Edge, Config

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

    #save_dir = "PaulD/checkpoint-14399"
    save_dir = "PaulD/IOA_261022-11999"
    config_class, model_class = (BertConfig, RelationPT)
    print("load ckpt from {}".format(save_dir))
    config = config_class.from_pretrained(save_dir)  # , num_labels = len(label_list))
    model = model_class.from_pretrained(save_dir, config=config)
    path = './processed/vocab.json'
    model.config.vocab = load_vocab(path)

    #n_gpu = torch.cuda.device_count()
    #if torch.cuda.is_available():  #
    #    model.cuda()
    #    if n_gpu > 1:
    #       model = torch.nn.DataParallel(model)
    return model

@st.experimental_singleton
def load_embeddings(input_dir, _model, device):
    print("loading embeddings")
    ## Loads embeddings
    #with open(os.path.abspath(input_dir + "entity/entity_embeddings_test.pt"), 'rb') as f:
    with open(os.path.abspath(input_dir + "entity/entity_embeddings.pt"), 'rb') as f:
        _model.entity_embeddings = pickle.load(f)

    # print(st.session_state.get('attribute_embeddings')==None)

    if st.session_state.get('attribute_embeddings') == None:
        argument_inputs = load_classes(input_dir + "attribute/attribute.pt", device)
        with torch.no_grad():
            attribute_embeddings = _model.bert(input_ids=argument_inputs['input_ids'],
                                              attention_mask=argument_inputs['attention_mask'],
                                              token_type_ids=argument_inputs['token_type_ids'])[1]
            #set_state_if_absent('attribute_embeddings', attribute_embeddings)
            st.session_state.attribute_embeddings = attribute_embeddings

    if st.session_state.get('concept_embeddings') == None:
        argument_inputs = load_classes(input_dir + "concept/concept.pt", device)
        with torch.no_grad():
            concept_embeddings = _model.bert(input_ids=argument_inputs['input_ids'],
                                            attention_mask=argument_inputs['attention_mask'],
                                            token_type_ids=argument_inputs['token_type_ids'])[1]
            #set_state_if_absent('concept_embeddings', concept_embeddings)
            st.session_state.concept_embeddings = concept_embeddings

    argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
    # with torch.no_grad():
    #     relation_embeddings = _model.bert(input_ids=argument_inputs['input_ids'],
    #                                      attention_mask=argument_inputs['attention_mask'],
    #                                      token_type_ids=argument_inputs['token_type_ids'])[1]

    if st.session_state.get('relation_embeddings') == None:
        argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
        with torch.no_grad():
            relation_embeddings = _model.bert(input_ids=argument_inputs['input_ids'],
                                             attention_mask=argument_inputs['attention_mask'],
                                             token_type_ids=argument_inputs['token_type_ids'])[1]
        #set_state_if_absent('relation_embeddings', relation_embeddings)
        st.session_state.relation_embeddings = relation_embeddings

@st.experimental_singleton
def load_tokenizer(path):
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(path, do_lower_case=False)
    return tokenizer

def one_hop(engine, entity_id, streamlit_state):
    #print(entity_id)
    #print(streamlit_state)
    nodes = streamlit_state.get('nodes')
    edges = streamlit_state.get('edges')

    def insert_newlines(string, every=64):
        lines = []
        for i in range(0, len(string), every):
            lines.append(string[i:i + every])
        return '\n'.join(lines)

    def append_node(nodes, node):
        if not node.id in [n.id for n in nodes]:
            nodes.append(node)
        else:
            print('Skipping insertion of node')
            print(node.id)
        return nodes

    entity_inf = engine.kb.entities[entity_id[0]]
    node = Node(id=entity_id[0],label=entity_inf['name'],
                      size=25,shape="diamond")
    nodes = append_node(nodes, node)
    for count, att in enumerate(entity_inf['attributes']):
        node = Node(id=entity_id[0] + str(count),
                          label=att['key']+ ": " +insert_newlines(str(att['value'].value)[0:200],every=64),
                          size=10,
                          color="318ce7",
                          shape="dot"
                          # "circularImage", image, circularImage, diamond, dot, star, triangle, triangleDown, hexagon, square and icon
                          # image="http://mar)
                          )
        nodes = append_node(nodes, node)

        edges.append(Edge(source=entity_id[0] + str(count),
                          target=entity_id[0],
                          color="FDD2BS"))

    for count, rel in enumerate(entity_inf['relations']):
        #print(engine.kb.entities.get(rel['object'].split('/resource/')[1]))
        if engine.kb.entities.get(rel['object'].split('/resource/')[1]) != None:
             #    #engine.kb.entities[ ['name']
             node = Node(id=rel['object'].split('/resource/')[1],
                               label=engine.kb.entities[rel['object'].split('/resource/')[1]]['name'],
                               size=25,
                               color="318ce7",
                               shape="diamond"
                               )

             nodes = append_node(nodes, node)

        #
             edges.append(Edge(source=entity_id[0],
                               target = engine.kb.entities[rel['object'].split('/resource/')[1]]['name'],
                               color = "003153",
                               label = rel['relation']
                               ))

    return nodes, edges


DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "What is the mass of COS B?")

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def main():
    ################################################################
    ##### Load KG engine
    path = './Preprocessing_KG/esa_kb.json' #
    engine = load_kg(os.path.abspath(path))
    ################################################################

    ################################################################
    ##### Load Model
    # save_dir = '/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-14399'
    tokenizer =load_tokenizer('bert-base-cased')

    path = './processed/vocab.json'
    vocab = load_vocab(path)
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = './processed/'
    model = load_model()
    argument_inputs = load_classes(input_dir + "entity/entity_small.pt", device)

    # ## Todo needs replacement
    #argument_inputs = load_classes(input_dir + "entity/entity.pt", device)
    ents = tokenizer.batch_decode(argument_inputs['input_ids'], skip_special_tokens=True)
    vocab['entity2id'] = {}
    for entity in ents:
        if not entity in vocab['entity2id']:
            vocab['entity2id'][entity] = len(vocab['entity2id'])
    vocab['id2entity'] = [entity for entity, iid in vocab['entity2id'].items()]
    model.config.vocab = vocab

    ################################################################

    #st.set_page_config(layout='wide')

    ################################################################
    ## Persistent state for streamlit
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("results", {})
    set_state_if_absent('attribute_embeddings', None)
    set_state_if_absent('relation_embeddings', None)
    set_state_if_absent('concept_embeddings', None)

    ## KG visualisation
    set_state_if_absent("nodes", [])
    set_state_if_absent("edges", [])
    ################################################################


    def reset_results(*args):
        st.session_state.nodes = []
        st.session_state.edges = []
        #st.session_state.kg_output = None
        #st.session_state.changed = False
        st.session_state.results = {}


    # argument_inputs = load_classes(input_dir + "entity/entity.pt", device)
    # with torch.no_grad():
    #     model.entity_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
    #                                    attention_mask=argument_inputs['attention_mask'],
    #                                    token_type_ids=argument_inputs['token_type_ids'])[1]

    # ## Loads embeddings
    # with open(os.path.abspath(input_dir + "entity/entity_embeddings.pt"), 'rb') as f:
    #     model.entity_embeddings = pickle.load(f)
    #
    #
    # #print(st.session_state.get('attribute_embeddings')==None)
    #
    # if st.session_state.get('attribute_embeddings') == None:
    #     argument_inputs = load_classes(input_dir + "attribute/attribute.pt", device)
    #     with torch.no_grad():
    #         attribute_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
    #                                           attention_mask=argument_inputs['attention_mask'],
    #                                           token_type_ids=argument_inputs['token_type_ids'])[1]
    #         set_state_if_absent('attribute_embeddings', attribute_embeddings)
    #
    # if st.session_state.get('concept_embeddings') == None:
    #     argument_inputs = load_classes(input_dir + "concept/concept.pt", device)
    #     with torch.no_grad():
    #         concept_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
    #                                         attention_mask=argument_inputs['attention_mask'],
    #                                         token_type_ids=argument_inputs['token_type_ids'])[1]
    #         set_state_if_absent('concept_embeddings', concept_embeddings)
    #
    # argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
    # with torch.no_grad():
    #     relation_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
    #                                      attention_mask=argument_inputs['attention_mask'],
    #                                      token_type_ids=argument_inputs['token_type_ids'])[1]
    #
    # if st.session_state.get('relation_embeddings') == None:
    #     argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
    #     with torch.no_grad():
    #         relation_embeddings = model.bert(input_ids=argument_inputs['input_ids'],
    #                                          attention_mask=argument_inputs['attention_mask'],
    #                                          token_type_ids=argument_inputs['token_type_ids'])[1]
    #     set_state_if_absent('relation_embeddings', relation_embeddings)
    load_embeddings(input_dir, model,device)

    st.title("Demo for querying KG with ProgramTransfer method")
    st.markdown(""" *Note: do not use keywords, but full-fledged questions.*""")
    query = st.text_input('Enter a query for the KG', DEFAULT_QUESTION_AT_STARTUP, on_change=reset_results)
                            #"engine.QueryAttr(engine.Find('GAIA'),'Mass')")
    st.session_state.results['question'] = query
    inputs = tokenizer([query], return_tensors='pt', padding=True)
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
    st.session_state.results['program'] = object
    st.write(f"Predicted program: {str(object)}")

    #object = eval(object)
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

    ################################################################
    ## Generate KG visualisation
    #entities
    #set_state_if_absent("input", )  # DEFAULT_INPUT_AT_STARTUP
    #set_state_if_absent("kg_output", None)
    results = []
    # for function in object:
    #     if function['function'] == 'Find':
    #         st.session_state.results['entities'] =  st.session_state.results.get('entities',[]) + [function['inputs']]
    # st.json(st.session_state.results)
    # print(st.session_state.results)
    # if st.session_state.results.get('entities') != None:
    #     for entity in st.session_state.results['entities']:
    #         entity_id = engine.kb.name_to_id[entity[0]]
    #         st.session_state.nodes, st.session_state.edges = one_hop(engine, entity_id, st.session_state)
    #
    # # set_state_if_absent('kg_output', st.session_state.results['entities'])
    #
    # config = Config(width=900,
    #                 height=900,
    #                 # **kwargs
    #                 )
    #
    # kg_output = agraph(nodes=st.session_state['nodes'],
    #                    edges=st.session_state['edges'],
    #                    config=config)






main()