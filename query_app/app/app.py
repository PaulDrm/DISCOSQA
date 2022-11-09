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
from streamlit_agraph import agraph, Node, Edge, Config
import re
from copy import deepcopy

import altair as alt
import pandas as pd

from dateutil import parser
from sutime import SUTime

def predict_year(sentence, sutime):
    """
    predict year from sentence with Python wrapper of SUTime
    sentence:string : phrase which contains a year
    """

    parse = sutime.parse(sentence)
    parse = parser.parse(parse[0]['value']).year

    return parse

def predict_date(sentence,sutime):
    """
    predict year from sentence with Python wrapper of SUTime
    sentence:string : phrase which contains a year
    """

    parse = sutime.parse(sentence)
    return parse#['value']

@st.experimental_singleton
def load_sftipars():
    """
    Returns: Stanford time parser
    """
    sutime = SUTime(mark_time_ranges=True, include_range=True)
    return sutime

def predict_num(sentence):
    """
    get number from sentence with regular expression pattern
    sentence:string : phrase which contains a number
    """

    num = re.findall(r'\d+', sentence)
    return num


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
    """
    Transforms predicted program from neural networks
    into form that can be excecuted on KOPL database
    [{'function': 'Find', 'inputs': ['Gaia'], 'dependencies':[]},{'function': 'QueryAttr', 'inputs': ['CosparID'], 'dependencies':[1]}]
    -->
    engine.QueryAttr(engine.Find('Gaia'), 'CosparID'))
    Args:
        program: prediction of RelationPT model

    Returns: string: return KOPL program in string form that can be excecuted on the KOPLEngine class with eval(

    """

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
    #save_dir = "PaulD/IOA_261022-11999"
    save_dir = "PaulD/IOA_ft_07112022-33"
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

#@st.experimental_singleton
def load_embeddings(input_dir, _model, device):
    print("loading embeddings")
    ## Loads embeddings

    if _model.entity_embeddings == None:

        #with open(os.path.abspath(input_dir + "entity/entity_embeddings.pt"), 'rb') as f:
        #with open(os.path.abspath(input_dir + "entity/entity_embeddings_test.pt"), 'rb') as f:
        #with open(os.path.abspath(input_dir + "entity/entity_embeddings_3110.pt"), 'rb') as f:
        with open(os.path.abspath(input_dir + "entity/entity_embeddings_0711.pt"), 'rb') as f:

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

    ##argument_inputs = load_classes(input_dir + "relation/relation.pt", device)
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


def one_hop(engine, entity_ids, streamlit_state):
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

    for entity_id in entity_ids:
        entity_inf = engine.kb.entities[entity_id]
        #st.json(entity_inf)
        node = Node(id=entity_id,label=entity_inf['name'],
                          size=25,shape="diamond")
        nodes = append_node(nodes, node)
        for count, att in enumerate(entity_inf['attributes']):
            node = Node(id=entity_id + str(count),
                              label=att['key']+ ": " +insert_newlines(str(att['value'].value)[0:200],every=64),
                              size=10,
                              color="318ce7",
                              shape="dot"
                              # "circularImage", image, circularImage, diamond, dot, star, triangle, triangleDown, hexagon, square and icon
                              # image="http://mar)
                              )
            nodes = append_node(nodes, node)

            edges.append(Edge(source=entity_id + str(count),
                              target=entity_id,
                              color="FDD2BS"))

        for count, rel in enumerate(entity_inf['relations']):

            if engine.kb.entities.get(rel['object']) != None:
                #    #engine.kb.entities[ ['name']
                node = Node(id=rel['object'],
                            label=engine.kb.entities[rel['object']]['name'],
                            size=25,
                            color="318ce7",
                            shape="diamond"
                            )
                if len(nodes)< 50:
                    nodes = append_node(nodes, node)

                    #print(entity_id[0])
                    edges.append(Edge(source=entity_id,
                                      target=rel['object'],
                                      color="003153",
                                      label=rel['relation']
                                      ))
                else:
                    st.write(f"Visualisation of relations for {engine.kb.entities[entity_id]['name']} was limited to 50 to increase performance")
                    break

    return nodes, edges

def log_data():

    with open("./query_results/queries.json", 'r') as f:
        dataset = json.load(f)

    dataset.append(st.session_state.results)
    with open("./query_results/queries.json", 'w+') as f:
        json.dump(dataset, f)

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


    ##### Load Model
    # save_dir = '/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-14399'
    tokenizer =load_tokenizer('bert-base-cased')

    path = './processed/vocab.json'
    vocab = load_vocab(path)
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = './processed/'
    model = load_model()
    sutime = load_sftipars()
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
    set_state_if_absent('program', '')

    ## Other functionalities
    set_state_if_absent('feedback', False)
    set_state_if_absent('comment', '')

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
        st.session_state.feedback = False
        st.session_state.comment = ''
        st.session_state.program = []


    load_embeddings(input_dir, model,device)

    ### Train dataset for random questions
    with open("./test_data/IOA_test.json", 'r') as f:
        df = pd.read_json(f)



    st.title("Demo for querying KG with ProgramTransfer method")
    st.markdown(""" *Note: do not use keywords, but full-fledged questions.*""")
    # Enter a prediction of a model for querying the KB
    query = st.text_input('Enter a query for the KG', value=st.session_state.question, on_change=reset_results)
    print(query)

    col1, col2 = st.columns(2)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    # Get next random question from the CSV
    if col2.button("Random question"):
        reset_results()
        new_row = df.sample(1)
        while (
                new_row["question"].values[0] == st.session_state.question
        ):  # Avoid picking the same question twice (the change is not visible on the UI)
            new_row = df.sample(1)
        st.session_state.question = new_row["question"].values[0]
        st.session_state.answer = new_row["program"].values[0]
        st.session_state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        #raise st.scriptrunner.script_runner.RerunException(st.scriptrunner.script_requests.RerunData(None))
        st.experimental_rerun()
    st.session_state.random_question_requested = False

    run_query = (
                  run_pressed or query != st.session_state.question   #st.session_state.results['query']
                ) and not st.session_state.random_question_requested

    #st.session_state.results['question'] = query

    # Get results for query
    if run_query and query:
        reset_results()
        st.session_state.question = query

        ## Get prediction
        inputs = tokenizer([query], return_tensors='pt', padding=True)
        inputs = {key: inputs[key].to(device) for key in inputs.keys()}
        object, outputs = model.demo(**inputs, relation_embeddings=st.session_state.relation_embeddings, concept_embeddings=st.session_state.concept_embeddings,attribute_embeddings=st.session_state.attribute_embeddings)#[0]

        ## Transform prediction
        object = object[0]
        #outputs = outputs[0]
        ##### Dependencies
        for count, function in enumerate(object):
            if count == 0:
                function.update({'dependencies': []})
            else:
                function.update({'dependencies': [-1 + count]})


        ## Predict values
        ##### Time values
        for count, function in enumerate(object):
            if function['function'] == 'FilterYear':
                #function['inputs'] = function['inputs'] + [predict_year(query)]
                function['inputs'].insert(1, str(predict_year(query, sutime)))
            elif function['function'] == 'FilterDate':
                #function['inputs'] = function['inputs'] + [predict_date(query)]
                function['inputs'].insert(1, str(predict_date(query, sutime)[0]['value']))
        ##### Number values
            elif function['function'] == 'FilterNum':
                #function['inputs'] = function['inputs'] + [predict_num(query)]
                function['inputs'].insert(1, str(predict_num(query)[0]))

            elif function['function'] == 'Relate':
                function['inputs'].append('forward')

        st.session_state.results['question'] = query
        st.session_state.results['program'] = object

        ###############################################################
        #### MODIFY PROGRAM FOR TOPK-return
        ###############################################################
        mod = deepcopy(st.session_state.results['program'])
        key = 'topk_entities'
        for num_batch, parse in enumerate([mod]):

            ## for each argument in prediction
            for argument_num, pos in enumerate(outputs[key][0][num_batch]):
                ## if argument is pad --> skip
                if pos != 0:
                    ## add argument to right function
                    parse[pos - 1]['inputs'] = parse[pos - 1]['inputs'] + [outputs[key][1][num_batch][argument_num]]

        st.session_state.results['topk_results'] = mod#[0]
        st.session_state.results['time'] = str(time.time())
        log_data()

        #st.write(st.session_state.results)

    #object = eval(object)
    #object
    # try:
    #     results = eval(parse_program(object))
    #     if len(result)==1:
    #         st.write(result[0].value)


    ###############################################################
    ## FEEDBACK
    ###############################################################


    comment = st.text_area('Additional comment for feedback',"", key = "comment")



    def clear_text():
        st.session_state.results['comment'] = comment
        st.session_state.comment = ""

    button_col1, button_col2, _ = st.columns([1, 1, 6])
    button1 = button_col1.button("👍", help="Correct answer",  on_click=clear_text)
    button2 = button_col2.button("👎", help="Wrong answer", on_click=clear_text)


    if button1 and not st.session_state.feedback:  # key=f"{result['context']}{count}1",
        print('saving results...')
        st.write("Thanks for your feedback!")
        print(st.session_state.results)
        #st.session_state.comment = ""
        st.session_state.results['time'] = str(time.time())
        st.session_state.results['label'] = "True"
        st.session_state.feedback = True
        log_data()

    elif button1:
        st.write("Feedback already received. Please try another question")

    if button2 and not st.session_state.feedback:  # key=f"{result['context']}{count}2",
        print('saving results...')
        st.write("Thanks for your feedback!")
        #st.session_state.comment = ""
        st.session_state.results['time'] = str(time.time())
        st.session_state.results['label'] = "False"
        st.session_state.feedback = True
        log_data()

    elif button2:
        st.write("Feedback already received. Please try another question")

    ###############################################################
    ## GET ANSWERS AND VISUALISATION
    ###############################################################

    # Get results for query
    #if run_query and query:
    print(st.session_state.results.get('program'))
    if st.session_state.results.get('program') != None:

        st.caption("Results", unsafe_allow_html=False)

        filt_y = 0
        filt_n = 0
        vis = 0
        what_check = 0
        index_vis = []
        find_check = 0
        program = st.session_state.results.get('program')
        for idx, function in enumerate(program):
            if function['function'] == "What":
                program = program[:-1]
                vis = 1
                what_check = 1
            elif function['function'] == "FilterYear":
                filt_y = 1
                index_y = idx
                bin = 1
            elif function['function'] == "FilterNum":
                filt_n = 1
                index_n = idx

            elif function['function'] == "Find":
                vis = 1
                find_check = 1
                index_vis.append(idx)

        #if st.session_state.program != program:
        #    st.session_state.program = program

        results = eval(parse_program(program))
        st.write(f"Predicted program: {str(program)}")
        st.write(f'Results for the query "{query} are: ')
        st.session_state.results['answer'] = str(results)
        if program[-1]['function'] == 'QueryAttr':
            st.write([result.value for result in results])
        elif what_check:
            st.write([engine.kb.entities[id]['name'] for id in results[0]])
        #elif len(results) == 0:
        #    st.write("Results empty. Please try to rephrase question or submit feedback.")
        else:
            st.write(results)

        if program[-1]['function'] == 'Count' and filt_y:
            # if st.session_state.modification == None:
            index = index_y
            current = program
            programs = []
            programs.append(
                {current[index]['inputs'][0]: current[index]['inputs'][1], 'result': eval(parse_program(current))})
            # st.write(int(current[index]['inputs'][1]))
            if st.session_state.program[index]['inputs'][2] == '=':
                for i in range(1, 6, bin):
                    temp = deepcopy(current)  # [index]['inputs'
                    value1 = str(int(temp[index]['inputs'][1]) + i)
                    value2 = str(int(temp[index]['inputs'][1]) - i)
                    temp[index]['inputs'][1] = value1
                    programs.append({temp[index]['inputs'][0]: value1, 'result': eval(
                        parse_program(deepcopy(temp)))})  # 'result': eval(parse_program(deepcopy(temp)))})
                    temp[index]['inputs'][1] = value2
                    programs.append({temp[index]['inputs'][0]: value2, 'result': eval(parse_program(deepcopy(temp)))})
                df = pd.DataFrame(programs)
                df = df.sort_values(by=temp[index]['inputs'][0], ascending=True)
                st.write(df)
                # st.bar_chart())

                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(temp[index]['inputs'][0], sort=None),
                    y=df.columns[1:].values[0],
                    color=alt.condition(
                        alt.datum[temp[index]['inputs'][0]] == current[index]['inputs'][1],
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            elif current[index]['inputs'][2] == '>':

                for idx, function in enumerate(current):
                    if function['function'] == "FilterConcept":
                        conc = engine.Find(function['inputs'][0])
                        values = engine.kb.concept_key_values[conc[0][0]][current[index]['inputs'][0]]
                        values = [int(item.value.year) for item in values]

                for i in range(int(current[index]['inputs'][1]) + bin, max(values), bin):
                    temp = deepcopy(current)  # [index]['inputs'
                    value1 = str(i)
                    # value2 = str(int(temp[index]['inputs'][1]) - i)
                    temp[index]['inputs'][1] = value1
                    programs.append({temp[index]['inputs'][0]: value1, 'result': eval(
                        parse_program(deepcopy(temp)))})  # 'result': eval(parse_program(deepcopy(temp)))})
                    # temp[index]['inputs'][1] = value2
                    # programs.append({temp[index]['inputs'][0]: value2, 'result': eval(parse_program(deepcopy(temp)))})
                df = pd.DataFrame(programs)
                st.write(df)
                # st.bar_chart(df.set_index(temp[index]['inputs'][0]))

                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(temp[index]['inputs'][0], sort=None),
                    y=df.columns[1:].values[0],
                    color=alt.condition(
                        alt.datum[temp[index]['inputs'][0]] == current[index]['inputs'][1],
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            elif current[index]['inputs'][2] == '<':

                for idx, function in enumerate(current):
                    if function['function'] == "FilterConcept":
                        conc = engine.Find(function['inputs'][0])
                        values = engine.kb.concept_key_values[conc[0][0]][current[index]['inputs'][0]]
                        values = [item.value.year for item in values]
                        # st.write(min(values))
                for i in range(min(values), int(current[index]['inputs'][1]), bin):
                    # print(i)
                    temp = deepcopy(current)  # [index]['inputs'
                    value1 = str(int(i))
                    # value2 = str(int(temp[index]['inputs'][1]) - i)
                    temp[index]['inputs'][1] = value1
                    programs.append({temp[index]['inputs'][0]: value1, 'result': eval(
                        parse_program(deepcopy(temp)))})  # 'result': eval(parse_program(deepcopy(temp)))})
                    # temp[index]['inputs'][1] = value2
                    # programs.append({temp[index]['inputs'][0]: value2, 'result': eval(parse_program(deepcopy(temp)))})
                df = pd.DataFrame(programs)

                # st.bar_chart(df.set_index(temp[index]['inputs'][0]))
                df = df.sort_values(by=temp[index]['inputs'][0], ascending=True)
                st.write(df)
                # df.set_index(temp[index]['inputs'][0]
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(temp[index]['inputs'][0], sort=None),
                    y=df.columns[1:].values[0],
                    color=alt.condition(
                        alt.datum[temp[index]['inputs'][0]] == current[index]['inputs'][1],
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

        if program[-1]['function'] == 'Count' and filt_n:
            # if st.session_state.modification == None:
            index = index_n
            current = program
            programs = []
            programs.append(
                {current[index]['inputs'][0]: int(current[index]['inputs'][1]), 'result': eval(parse_program(current))})
            # st.write(float(current[index]['inputs'][1])),
            # bin = float(current[index]['inputs'][1]) // 10

            control = 0
            for idx, function in enumerate(current):
                if function['function'] == "FilterConcept":
                    conc = engine.Find(function['inputs'][0])
                    values = engine.kb.concept_key_values[conc[0][0]][current[index]['inputs'][0]]
                    values = [item.value for item in values]
                    control = 1

            if control == 0:
                ### TODO adapt the pipeline possibly produce error and send feedback
                pass

            if st.session_state.program[index]['inputs'][2] == '=':

                for idx, function in enumerate(current):
                    if function['function'] == "FilterConcept":
                        conc = engine.Find(function['inputs'][0])
                        values = engine.kb.concept_key_values[conc[0][0]][current[index]['inputs'][0]]
                        values = [float(item.value) for item in values]

                starting_point = values.index(float(current[index]['inputs'][1]))
                # for i in range(bin, int(6*bin),int(bin)):
                #     temp = deepcopy(current)  # [index]['inputs'
                #     value1 = str(int(temp[index]['inputs'][1]) + i)
                #     value2 = str(int(temp[index]['inputs'][1]) - i)
                #     temp[index]['inputs'][1] = value1
                #     programs.append({temp[index]['inputs'][0]: value1, 'result': eval(parse_program(deepcopy(temp)))})#'result': eval(parse_program(deepcopy(temp)))})
                #     temp[index]['inputs'][1] = value2
                #     programs.append({temp[index]['inputs'][0]: value2, 'result': eval(parse_program(deepcopy(temp)))})
                st.write(starting_point)
                for i in range(1, 6):
                    temp = deepcopy(current)  # [index]['inputs'
                    if starting_point + i < len(values):
                        value1 = str(values[starting_point + i])
                        temp[index]['inputs'][1] = value1
                        programs.append({temp[index]['inputs'][0]: int(float(value1)), 'result': eval(
                            parse_program(deepcopy(temp)))})  # 'result': eval(parse_program(deepcopy(temp)))})
                    if starting_point - i > 0:
                        value2 = str(values[starting_point - i])
                        temp[index]['inputs'][1] = value2
                        programs.append(
                            {temp[index]['inputs'][0]: int(float(value2)), 'result': eval(parse_program(deepcopy(temp)))})

                df = pd.DataFrame(programs)
                df = df.sort_values(by=temp[index]['inputs'][0], ascending=True)
                st.write(df)
                # df.dtypes
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(temp[index]['inputs'][0], sort=None, type='nominal'),
                    y=df.columns[1:].values[0],
                    color=alt.condition(
                        alt.datum[temp[index]['inputs'][0]] == int(current[index]['inputs'][1]),
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

                # st.bar_chart(df.set_index(temp[index]['inputs'][0]))
            elif current[index]['inputs'][2] == '>':

                for idx, function in enumerate(current):
                    if function['function'] == "FilterConcept":
                        conc = engine.Find(function['inputs'][0])
                        values = engine.kb.concept_key_values[conc[0][0]][current[index]['inputs'][0]]
                        values = [item.value for item in values]

                for i in range(int(current[index]['inputs'][1]) + int(bin), int(max(values)), int(bin)):
                    # print(i)
                    temp = deepcopy(current)  # [index]['inputs'
                    value1 = str(i)
                    # value2 = str(int(temp[index]['inputs'][1]) - i)
                    temp[index]['inputs'][1] = value1
                    programs.append({temp[index]['inputs'][0]: int(float(value1)), 'result': eval(
                        parse_program(deepcopy(temp)))})  # 'result': eval(parse_program(deepcopy(temp)))})
                    # temp[index]['inputs'][1] = value2
                    # programs.append({temp[index]['inputs'][0]: value2, 'result': eval(parse_program(deepcopy(temp)))})
                df = pd.DataFrame(programs)
                # df.sort_values(by=temp[index]['inputs'][0], ascending=True)
                st.write(df)
                # st.bar_chart(df.set_index(temp[index]['inputs'][0]))
                # print(df.columns[1:])
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(temp[index]['inputs'][0], sort=None, type='nominal'),
                    y=df.columns[1:].values[0],
                    color=alt.condition(
                        alt.datum[temp[index]['inputs'][0]] == int(current[index]['inputs'][1]),
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            elif current[index]['inputs'][2] == '<':

                # for idx, function in enumerate(current):
                #     if function['function'] == "FilterConcept":
                #         conc = engine.Find(function['inputs'][0])
                #         values = engine.kb.concept_key_values[conc[0][0]][current[index]['inputs'][0]]
                #         values = [item.value for item in values]
                #         #st.write(min(values))
                bin = int(current[index]['inputs'][1]) // 10
                for i in range(0, int(current[index]['inputs'][1]), int(bin)):
                    # print(i)
                    temp = deepcopy(current)  # [index]['inputs'
                    value1 = str(int(i))
                    # value2 = str(int(temp[index]['inputs'][1]) - i)
                    temp[index]['inputs'][1] = value1
                    programs.append({temp[index]['inputs'][0]: int(float(value1)), 'result': eval(
                        parse_program(deepcopy(temp)))})  # 'result': eval(parse_program(deepcopy(temp)))})
                    # temp[index]['inputs'][1] = value2
                    # programs.append({temp[index]['inputs'][0]: value2, 'result': eval(parse_program(deepcopy(temp)))})
                df = pd.DataFrame(programs)
                df = df.sort_values(by=temp[index]['inputs'][0], ascending=True)
                st.write(df)

                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(temp[index]['inputs'][0], sort=None, type='nominal'),
                    y=df.columns[1:].values[0],
                    color=alt.condition(
                        alt.datum[temp[index]['inputs'][0]] == int(current[index]['inputs'][1]),
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
                st.altair_chart(chart, use_container_width=True)

        if vis:
            entity_ids = []
            if what_check:
                entity_ids = entity_ids + results[0]

            if find_check:
                for idx in index_vis:
                    entity_ids = entity_ids + engine.kb.name_to_id[program[idx]['inputs'][0]]

            st.session_state.nodes, st.session_state.edges = one_hop(engine, entity_ids, st.session_state)

            config = Config(width=900,
                            height=900,
                            # **kwargs
                            )

            kg_output = agraph(nodes=st.session_state['nodes'],
                               edges=st.session_state['edges'],
                               config=config)

    # if st.session_state.input != input: #st.session_state.get("kg_output") == None and
    #     st.session_state['input'] = input
    #     entity_id = engine.kb.name_to_id[input]
    #     st.write(entity_id)
    #     #nodes, edges = one_hop(engine, entity_id, st.session_state)
    #     st.session_state.nodes, st.session_state.edges = one_hop(engine, entity_id, st.session_state)
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