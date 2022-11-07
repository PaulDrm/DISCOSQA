from SPARQLWrapper import SPARQLWrapper, JSON
import json
import time
from tqdm import tqdm
from copy import deepcopy
def main():

    ## is slower
    # sparql = SPARQLWrapper(
    #     "http://localhost:7200/repositories/IOA_objects"
    # )

    sparql = SPARQLWrapper(
        "http://127.0.0.1:7200/repositories/IOA_objects"
    )
    sparql.setReturnFormat(JSON)

    #via a SPARQL endpoint
    sparql.setQuery("""PREFIX pred: <http://www.semanticweb.org/esa-ioa/ontologies/2022/predicates-ontology#>
                    select distinct ?subj where { ?subj pred:instance_of ?obj .}
                    """
    )
    # via a SPARQL endpoint
    # sparql.setQuery("""PREFIX pred: <http://www.semanticweb.org/esa-ioa/ontologies/2022/predicates-ontology#>
    #                    PREFIX ioa: <http://www.semanticweb.org/esa-ioa/ontologies/2022/ioa-wiki-ontology#>
    #                 select distinct ?subj where { ?subj pred:instance_of ioa:Object .}
    #                 """
    #                 )

    entity_extract = sparql.queryAndConvert()

    entities = {}
    concepts_names = {}

    print("entity attribute extraction")
    start_extract = time.time()
    for extract in tqdm(entity_extract['results']['bindings']):

        if '/resource/' in extract['subj']['value']:
            # via a SPARQL endpoint
            sparql.setQuery(f"""
                    SELECT ?uri ?pred ?o
                 WHERE {{
                    ?uri ?pred ?o.
                    FILTER (?uri = <{extract['subj']['value']}>) 
                    }}
                    """
                            )
        else:
            print("Skippin following entity in extraction..")
            print(extract['subj']['value'])
            continue
        #     print(f"""
        #         SELECT ?uri ?pred ?o
        #      WHERE {{
        #         ?uri ?pred ?o.
        #         FILTER (?uri = <{entity['subj']['value']}>)
        #         }}
        #         """)
        # ret = sparql.queryAndConvert()
        if extract['subj']['value'].split("/resource/")[1] in entities:
            continue

        results = sparql.queryAndConvert()

        #entity = {}
        entity = {'relations': [], 'attributes': []}
        for r in results["results"]["bindings"]:

            ## attributes
            ## check that value is not entity ("/ressource") or the name "/predicates-ontology"
            if not ("/ioa-wiki-ontology#" in r['o']['value'] or "/resource/" in r['o']['value'] or "/predicates-ontology" in r['pred']['value'] or "genid" in r['o']['value']):
                # entity['attributes'] = entity.get('attributes',[]) +[{'key':r['pred']['value'].split('#')[1],'value': {'value':r['o']['value'], 'type':'string'}, 'qualifiers':{} }]

                ## datatype
                ## assigns different datatypes to attributes
                if r['o'].get('datatype') != None:  # .split('XMLSchema#')[1] == 'dateTime':##None:


                    if r['o'].get('datatype').split('XMLSchema#')[1] == 'dateTime':
                        entity['attributes'] = entity.get('attributes', []) + [{'key': r['pred']['value'].split('#')[1],
                                                                                'value': {'value': r['o']['value'],
                                                                                          'type': 'date'},
                                                                                'qualifiers': {}}]

                    elif r['o'].get('datatype', "").split('XMLSchema#')[1] == 'float':
                        entity['attributes'] = entity.get('attributes', []) + [{'key': r['pred']['value'].split('#')[1],
                                                                                'value': {
                                                                                    'value': float(r['o']['value']),
                                                                                    'type': 'quantity', 'unit': 'm'},
                                                                                'qualifiers': {}}]

                    elif r['o'].get('datatype', "").split('XMLSchema#')[1] == 'unsignedInt':
                        entity['attributes'] = entity.get('attributes', []) + [{'key': r['pred']['value'].split('#')[1],
                                                                                'value': {'value': r['o']['value'],
                                                                                          'type': 'string'},
                                                                                'qualifiers': {}}]

                    elif r['o'].get('datatype', "").split('XMLSchema#')[1] == 'boolean':
                        entity['attributes'] = entity.get('attributes', []) + [{'key': r['pred']['value'].split('#')[1],
                                                                                'value': {'value': r['o']['value'],
                                                                                          'type': 'string'},
                                                                                'qualifiers': {}}]

                    elif r['o'].get('datatype', "").split('XMLSchema#')[1] == 'nonNegativeInteger':
                        entity['attributes'] = entity.get('attributes',[]) +[{'key':r['pred']['value'].split('#')[1],'value': {'value':float(r['o']['value']), 'type':'quantity', 'unit':'m'}, 'qualifiers':{} }]




                    else:
                        print("Datatype not recognised")
                        print(r['o'].get('datatype', ""))


                else:
                    entity['attributes'] = entity.get('attributes', []) + [
                        {'key': r['pred']['value'].split('#')[1], 'value': {'value': r['o']['value'], 'type': 'string'},
                         'qualifiers': {}}]


            ## relations
            elif "/resource/" in r['o']['value']:
                # entity['relations'] =  entity.get('relations',[]) +[{'predicate':r['pred']['value'].split('#')[1],'object': r['o']['value'],'direction':'forward', 'qualifiers':{} }]
                entity['relations'] = entity.get('relations', []) + [
                    {'relation': r['pred']['value'].split('#')[1], 'object': r['o']['value'].split('/resource/')[1], 'direction': 'forward',
                     'qualifiers': {}}]

            ## names
            elif "predicates-ontology#name" in r['pred']['value']:
                entity['name'] = r['o']['value']

            ## instance_of
            # elif "/ioa-wiki-ontology#" in r['o']['value']:
            elif "instance_of" in r['pred']['value']:

                if concepts_names.get(r['o']['value'].split('#')[1], None) == None:
                    concepts_names[r['o']['value'].split('#')[1]] = f'conc{str(len(concepts_names))}'
                entity['instanceOf'] = entity.get('instanceOf', []) + [concepts_names[r['o']['value'].split('#')[1]]]

        if entity.get('name')== None:
            entity['name'] = extract['subj']['value'].split("/resource/")[1]

        entities[extract['subj']['value'].split("/resource/")[1]] = entity
    print("Extraction time: ", time.time() - start_extract)

    # entities
    concepts = {iid: {'name': concept, 'instanceOf': []} for concept, iid in concepts_names.items()}
    kb_base = {'concepts': concepts, 'entities': entities}  # ['root'] =
    print('process entities')
    for eid, ent_info in tqdm(kb_base['entities'].items()):
        ### Todo changed here --> not all objects include relationships: rule to check beforehand
        # if kb['entities'][eid].get('relations') != None:
        for rel_info in kb_base['entities'][eid]['relations']:
            obj_id = rel_info['object']
            if obj_id in kb_base['entities']:
                rel_info_for_con = {
                    ## Todo changed here --> 'relation' to 'predicate' to get ino
                    # 'relation': rel_info['relation'],
                    'relation': rel_info['relation'],
                    'direction': 'forward',  # if rel_info['direction']=='backward' else 'backward',
                    'object': eid,
                    'qualifiers': deepcopy(rel_info['qualifiers']),
                }
                if rel_info_for_con not in kb_base['entities'][obj_id]['relations']:
                    kb_base['entities'][obj_id]['relations'].append(rel_info_for_con)



    with open('esa_kb.json', 'w') as fp:
        json.dump(kb_base, fp)

if __name__ == '__main__':
    main()

