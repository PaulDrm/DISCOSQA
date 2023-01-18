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
            sparql.setQuery(f"""PREFIX pred: <http://www.semanticweb.org/esa-ioa/ontologies/2022/predicates-ontology#>
                               PREFIX ioa: <http://www.semanticweb.org/esa-ioa/ontologies/2022/ioa-wiki-ontology#>
                               SELECT ?uri ?pred ?o ?value ?unit WHERE {{ 
                            ?uri ?pred ?o .
                            FILTER (?uri = <{extract['subj']['value']}>) 
                            OPTIONAL{{?o pred:value ?value .}}
                            OPTIONAL{{ ?o pred:unit ?unit .}}
                            OPTIONAL{{ ?o pred:date ?value .}}
                        }} limit 100  """)

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
            if r['o']['type'] == "bnode" and "node" in r['o']['value']:

                ## datatype
                ## assigns different datatypes to attributes
                if r.get('value') != None:
                    if r['value'].get('datatype') != None:  # .split('XMLSchema#')[1] == 'dateTime':##None:

                        if r['value'].get('datatype').split('XMLSchema#')[1] == 'dateTime':
                            entity['attributes'] = entity.get('attributes', []) + [
                                {'key': r['pred']['value'].split('#')[1],
                                 'value': {'value': r['value']['value'],
                                           'type': 'date'},
                                 'qualifiers': {}}]

                        elif r['value'].get('datatype', "").split('XMLSchema#')[1] == 'float':
                            #if r['unit'].get('value') != None:
                            if r.get('unit') != None:
                                entity['attributes'] = entity.get('attributes', []) + [
                                    {'key': r['pred']['value'].split('#')[1],
                                     'value': {
                                         'value': round(float(r['value']['value']), 3),
                                         'type': 'quantity', 'unit': r['unit']['value']},
                                     'qualifiers': {}}]
                            else:
                                #print(r)
                                #print(extract['subj']['value'])
                                entity['attributes'] = entity.get('attributes', []) + [
                                    {'key': r['pred']['value'].split('#')[1],
                                     'value': {
                                         'value': round(float(r['value']['value']), 3),
                                         'type': 'quantity', 'unit': "-"},
                                     'qualifiers': {}}]

                        elif r['value'].get('datatype', "").split('XMLSchema#')[1] == 'unsignedInt':
                            entity['attributes'] = entity.get('attributes', []) + [
                                {'key': r['pred']['value'].split('#')[1],
                                 'value': {'value': r['value']['value'],
                                           'type': 'string'},
                                 'qualifiers': {}}]

                        elif r['value'].get('datatype', "").split('XMLSchema#')[1] == 'boolean':
                            entity['attributes'] = entity.get('attributes', []) + [
                                {'key': r['pred']['value'].split('#')[1],
                                 'value': {'value': r['value']['value'],
                                           'type': 'string'},
                                 'qualifiers': {}}]

                        elif r['value'].get('datatype', "").split('XMLSchema#')[1] == 'nonNegativeInteger':
                             entity['attributes'] = entity.get('attributes', []) + [
                                 {'key': r['pred']['value'].split('#')[1],
                                  'value': {'value': float(r['value']['value']), 'type': 'quantity', 'unit': 'n'},
                                  'qualifiers': {}}]
                            #print(extract['subj']['value'])

                        else:
                            print("Datatype not recognised")
                            print(r['o'].get('datatype', ""))


                    else:
                        entity['attributes'] = entity.get('attributes', []) + [
                            {'key': r['pred']['value'].split('#')[1],
                             'value': {'value': r['value']['value'], 'type': 'string'},
                             'qualifiers': {}}]

            ## relations
            elif "/resource/" in r['o']['value']:
                # entity['relations'] =  entity.get('relations',[]) +[{'predicate':r['pred']['value'].split('#')[1],'object': r['o']['value'],'direction':'forward', 'qualifiers':{} }]

                # if r['pred']['value'].split('#')[1] == "initial_orbit" or r['pred']['value'].split('#')[1] == "destination_orbit":
                if r['pred']['value'].split('#')[1] == "initial_orbit" or r['pred']['value'].split('#')[
                    1] == "destination_orbit":
                    # print('lol')
                    entity['relations'] = entity.get('relations', []) + [
                        {'relation': 'orbit', 'object': r['o']['value'].split('/resource/')[1], 'direction': 'forward',
                         'qualifiers': {}}]

                if r['pred']['value'].split('#')[1] == "initial_orbit" or r['pred']['value'].split('#')[
                    1] == "destination_orbit":
                    # print('lol')
                    entity['relations'] = entity.get('relations', []) + [
                        {'relation': 'orbit', 'object': r['o']['value'].split('/resource/')[1], 'direction': 'forward',
                         'qualifiers': {}}]

                elif r['pred']['value'].split('#')[1] == "entity":
                    # print('lol')
                    entity['relations'] = entity.get('relations', []) + [
                        {'relation': 'operator', 'object': r['o']['value'].split('/resource/')[1],
                         'direction': 'forward', 'qualifiers': {}}]

                elif r['pred']['value'].split('#')[1] == "host_country":
                    # print('lol')
                    entity['relations'] = entity.get('relations', []) + [
                        {'relation': 'state', 'object': r['o']['value'].split('/resource/')[1], 'direction': 'forward',
                         'qualifiers': {}}]

                else:
                    entity['relations'] = entity.get('relations', []) + [
                        {'relation': r['pred']['value'].split('#')[1], 'object': r['o']['value'].split('/resource/')[1],
                         'direction': 'forward',
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

        if entity.get('name') == None:
            entity['name'] = extract['subj']['value'].split("/resource/")[1]

        for att in entity['attributes']:
            if att['key'] == 'reentry':
                for rel in entity['relations']:
                    if rel['relation'] == 'orbit':
                        rel['relation'] = "historic_orbit"


        entities[extract['subj']['value'].split("/resource/")[1]] = entity
    print("Extraction time: ", time.time() - start_extract)

    # entities
    #concepts = {iid: {'name': concept, 'instanceOf': []} for concept, iid in concepts_names.items()}

    concepts = {"conc0": {"name": "Object", "instanceOf": []}
              , "conc1": {"name": "RocketBodyObjClass", "instanceOf": ["conc0"]}
              , "conc2": {"name": "Entity", "instanceOf": []}
              , "conc3": {"name": "Organisation", "instanceOf": ["conc2"]}
              , "conc4": {"name": "Country", "instanceOf": ["conc2"]}
              , "conc5": {"name": "PayloadObjClass", "instanceOf": ["conc0"]}
              , "conc6": {"name": "PayloadMissionRelatedObjectObjClass", "instanceOf": ["conc0"]}
              , "conc7": {"name": "RocketMissionRelatedObjectObjClass", "instanceOf": ["conc0"]}
              , "conc8": {"name": "RocketFragmentationDebrisObjClass", "instanceOf": ["conc0"]}
              , "conc9": {"name": "PayloadFragmentationDebrisObjClass", "instanceOf": ["conc0"]}
              , "conc10": {"name": "PayloadDebrisObjClass", "instanceOf": ["conc0"]}
              , "conc11": {"name": "RocketDebrisObjClass", "instanceOf": ["conc0"]}
              , "conc12": {"name": "UnknownObjClass", "instanceOf": ["conc0"]}
              , "conc13": {"name": "OtherDebrisObjClass", "instanceOf": ["conc0"]}
              , "conc14": {"name": "OtherMissionRelatedObjectObjClass", "instanceOf": ["conc0"]}
              , "conc15": {"name": "Fragmentation", "instanceOf": []}
              , "conc16": {"name": "Launch", "instanceOf": []}
              , "conc17": {"name": "LaunchSite", "instanceOf": []}
              , "conc18": {"name": "LaunchVehicle", "instanceOf": []}
              , "conc19": {"name": "LaunchVehicleEngine", "instanceOf": []}
              , "conc20": {"name": "LaunchVehicleFamily", "instanceOf": []}
              , "conc21": {"name": "LaunchVehicleStage", "instanceOf": []}
              , "conc22": {"name": "LaunchSystem", "instanceOf": []}
              , "conc23": {"name": "DestinationOrbit", "instanceOf": []}
              , "conc24": {"name": "LEO", "instanceOf": ["conc38"]}
              , "conc25": {"name": "MGO", "instanceOf": ["conc38"]}
              , "conc26": {"name": "GTO", "instanceOf": ["conc38"]}
              , "conc27": {"name": "HEO", "instanceOf": ["conc38"]}
              , "conc28": {"name": "LMO", "instanceOf": ["conc38"]}
              , "conc29": {"name": "MEO", "instanceOf": ["conc38"]}
              , "conc30": {"name": "NSO", "instanceOf": ["conc38"]}
              , "conc31": {"name": "EGO", "instanceOf": ["conc38"]}
              , "conc32": {"name": "GEO", "instanceOf": ["conc38"]}
              , "conc33": {"name": "HAO", "instanceOf": ["conc38"]}
              , "conc34": {"name": "GHO", "instanceOf": ["conc38"]}
              , "conc35": {"name": "IGO", "instanceOf": ["conc38"]}
              , "conc36": {"name": "Propellant", "instanceOf": []}
              , "conc37": {"name": "InitialOrbit", "instanceOf": []}
              , "conc38": {"name": "EarthOrbit", "instanceOf": []}}

    kb_base = {'concepts': concepts, 'entities': entities}  # ['root'] =
    print('process entities')
    for eid, ent_info in tqdm(kb_base['entities'].items()):
        ### Todo changed here --> not all objects include relationships: rule to check beforehand
        # if kb['entities'][eid].get('relations') != None:
        for rel_info in kb_base['entities'][eid]['relations']:
            obj_id = rel_info['object']
            if obj_id in kb_base['entities']:
                rel_info_for_con = {
                    ### Todo changed here --> 'relation' to 'predicate'
                    # 'relation': rel_info['relation'],
                    'relation': rel_info['relation'],
                    'direction': 'forward',  # if rel_info['direction']=='backward' else 'backward',
                    'object': eid,
                    'qualifiers': deepcopy(rel_info['qualifiers']),
                }
                if rel_info_for_con not in kb_base['entities'][obj_id]['relations']:
                    kb_base['entities'][obj_id]['relations'].append(rel_info_for_con)



    with open('esa_kb_remote.json', 'w') as fp:
        json.dump(kb_base, fp)

if __name__ == '__main__':
    main()

