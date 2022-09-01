"""Functions for converting OntoRefine mapping to wdt-style relations"""

import numpy as np
from pprint import pprint
from tqdm.notebook import trange
import json
import pdb


def f_prefix(d):
    return d['property']['transformation']['expression']

def f_property(d):
    return d['property']['valueSource']['constant']

def f_colname(d):
    return d['values'][0]['valueSource']['columnName']

def f_dtype(d):
    return d['values'][0]['valueType']['datatype']['valueSource']['constant']


def get_unique_bnode_dict(colname,propname,datatype,dataprefix='xsd',unit=None):
    """Return dictionary with new bnode properties"""
    prop =  {'transformation': {'expression': 'ioa', 'language': 'prefix'},
             'valueSource': {'constant': propname, 'source': 'constant'}}
    val_type = {'propertyMappings': [{'property': {'transformation': {'expression': 'pred', 'language': 'prefix'},
                                                   'valueSource': {'constant': 'value', 'source': 'constant'}},
                                      'values': [{'valueSource': {'columnName': colname, 'source': 'column'},
                                                  'valueType': {'datatype': {'transformation': {'expression': dataprefix, 
                                                                                                'language': 'prefix'},
                                                                             'valueSource': {'constant': datatype, 
                                                                                             'source': 'constant'}},
                                                                'type': 'datatype_literal'}}]}],
                'type': 'unique_bnode'}
    
    if unit is not None:
        val_type['propertyMappings'].append(
            {'property': {'transformation': {'expression': 'pred','language': 'prefix'},
                          'valueSource': {'source': 'constant', 'constant': 'unit'}},
              'values': [{'valueSource': {'source': 'constant', 'constant': unit},
                          'valueType': {'datatype': {'transformation': {'expression': 'xsd','language': 'prefix'},
                                                     'valueSource': {'source': 'constant', 'constant': 'string'}},
                                        'type': 'datatype_literal'}}]}
        )
    vals = [{'valueSource': {'columnName': 'Column', 'source': 'column'},
             'valueType': val_type}]
    return {'property':prop, 'values':vals}


def get_value_bnode_relation(s_name,p_name,o_name,grel_expr,qual_name=[],qual_col=[],qual_prefix=[],qual_dtype=[]):
    """Return dictionary with bnode relating entities and qualifiers"""
    
    values = {'transformation': {'expression': grel_expr, 'language': 'grel'},
               'valueSource': {'source': 'row_index'},
               'valueType': {'propertyMappings': [
                   {'property': {'transformation': {'expression': 'pred', 'language': 'prefix'},
                                 'valueSource': {'source': 'constant', 'constant': 'fact_r'}},
                    'values': [{'transformation': {'expression': 'ioa', 'language': 'prefix'},
                                'valueSource': {'source': 'constant', 'constant': p_name},
                                'valueType': {'propertyMappings': [],'type': 'iri','typeMappings': []}}]},
                   {'property': {'transformation': {'expression': 'pred','language': 'prefix'},
                                 'valueSource': {'source': 'constant', 'constant': 'fact_t'}},
                    'values': [{'valueSource': {'columnName': o_name,'source': 'column'},
                                'valueType': {'propertyMappings': [],'type': 'iri','typeMappings': []}}]},
                   {'property': {'transformation': {'expression': 'pred','language': 'prefix'},
                                 'valueSource': {'source': 'constant', 'constant': 'fact_h'}},
                    'values': [{'valueSource': {'columnName': s_name,'source': 'column'},
                                'valueType': {'propertyMappings': [],'type': 'iri','typeMappings': []}}]}],
                             'type': 'value_bnode'}}
    
    for qual, col, prefix, dtype in zip(qual_name,qual_col,qual_prefix,qual_dtype):
        values['valueType']['propertyMappings'].append(
            {'property': {'transformation': {'expression': prefix,'language': 'prefix'},
                          'valueSource': {'source': 'constant', 'constant': qual}},
             'values': [{'valueSource': {'columnName': col, 'source': 'column'},
                         'valueType': {'type': 'datatype_literal',
                                       'datatype': {'transformation': {'expression': 'xsd','language': 'prefix'},
                                                    'valueSource': {'source': 'constant','constant': dtype}}}}]},
        )
    
    prop_map = {'propertyMappings': [{'property': {'transformation': {'expression': 'ioa', 'language': 'prefix'},
                                                   'valueSource': {'source': 'constant', 'constant': p_name}},
                                      'values': [{'valueSource': {'columnName': o_name, 'source': 'column'},
                                                  'valueType': {'propertyMappings': [],'type': 'iri','typeMappings': []}}]},
                                     {'property': {'transformation': {'expression': 'ioa', 'language': 'prefix'},
                                                   'valueSource': {'source': 'constant', 'constant': 'hasRelation'}},
                                      'values': [values]}],
                'subject': {'valueSource': {'columnName': s_name, 'source': 'column'}},'typeMappings': []}
    return prop_map


def get_new_relation(s_name,p_name,o_name):
    """Returns a s - p - o relation between two column names"""
    return {'propertyMappings': [{'property': {'transformation': {'expression': 'ioa','language': 'prefix'},
                                               'valueSource': {'source': 'constant', 'constant': p_name}},
                                  'values': [{'valueSource': {'columnName': o_name, 'source': 'column'},
                                              'valueType': {'propertyMappings': [],'type': 'iri','typeMappings': []}}]}],
            'subject': {'valueSource': {'columnName': s_name, 'source': 'column'}},
            'typeMappings': []}


def all_blank_nodes(data_properties,prop_transform):
    """Transform all properties in data_properties to blank node relations"""
    blank_nodes = []
    for p_data in data_properties:
        prop_old = f_property(p_data)
        prefix = f_prefix(p_data)
        if prop_old in prop_transform and prefix=='ioa':
            colname = f_colname(p_data)
            datatype = f_dtype(p_data)
            propname = prop_transform[prop_old]
            blank_nodes.append( get_unique_bnode_dict(colname,propname,datatype) )
    return blank_nodes


def all_blank_nodes_units(data_properties,prop_transform,unit_transform):
    """Transform all properties in data_properties to blank node relations with units"""
    blank_nodes = []
    
    for p_data in data_properties:
        prop_old = f_property(p_data)
        prefix = f_prefix(p_data)
        
        if prop_old in prop_transform and prefix=='ioa':
            colname = f_colname(p_data)
            datatype = f_dtype(p_data)
            propname = prop_transform[prop_old]
            
            if prop_old in unit_transform:
                unit = unit_transform[prop_old]
            else:
                unit = None
                
            blank_nodes.append( get_unique_bnode_dict(colname,propname,datatype,unit=unit) )
            
    return blank_nodes


def all_blank_nodes_relations(data_properties,rel_transform):
    """Transform all relations in data_properties to new names with optional qualifications"""
    rel_nodes = []
    
    for p_data in data_properties:
        prop_old = f_property(p_data)
        prefix = f_prefix(p_data)
        
        if prop_old in rel_transform and prefix=='ioa':
            colname = f_colname(p_data)
            reldict = rel_transform[prop_old]
            
            s_name = 'Column' if reldict['direction']=='forward' else colname
            p_name = reldict['name']
            o_name = colname if reldict['direction']=='forward' else 'Column'
            
            if reldict['qual'] is None:
                rel_nodes.append( get_new_relation(s_name, p_name, o_name) )
            else:
                q = reldict['qual']
                rel_nodes.append( get_value_bnode_relation(s_name, p_name, o_name,
                                                           q['grel'],q['name'],q['column'],q['prefix'],q['dtype']) )
            
    return rel_nodes


def new_type_pred(type_map):
    """Return pred:instance_of for type map value source"""
    return {'property': {'transformation': {'expression': 'pred', 'language': 'prefix'},
                         'valueSource': {'constant': 'instance_of', 'source': 'constant'}},
            'values': [{'transformation': {'expression': 'ioa', 'language': 'prefix'},
                        'valueSource': type_map['valueSource'],
                        'valueType': {'propertyMappings': [],'type': 'iri','typeMappings': []}}]}

def new_name_pred(name_prop):
    """Return pred:name for name prop values"""
    return {'property': {'transformation': {'expression': 'pred', 'language': 'prefix'},
                         'valueSource': {'constant': 'name', 'source': 'constant'}},
            'values': name_prop['values']}



def new_properties(blank_nodes, fname_json, rel_nodes=[], prop_rm=[], pred_name=False):
    """Load mapping and extend properties with new blank nodes"""
    with open(fname_json, 'r') as f:
        data_map_new = json.load(f)
        
    prop_new = []
    for type_map in data_map_new['subjectMappings'][0]['typeMappings']:
        prop_new.append( new_type_pred(type_map) )
    
    for prop in data_map_new['subjectMappings'][0]['propertyMappings']:
        if f_property(prop) not in prop_rm:
            prop_new.append(prop)
        if f_property(prop)=='Name' and pred_name:
            prop_new.append( new_name_pred(prop) )
            
    data_map_new['subjectMappings'][0]['propertyMappings'] = prop_new
        
    data_map_new['subjectMappings'][0]['propertyMappings'].extend(blank_nodes)
    data_map_new['subjectMappings'].extend(rel_nodes)
    
    data_map_new['namespaces']['ioa'] = "http://www.semanticweb.org/esa-ioa/ontologies/2022/ioa-wiki-ontology#"
    data_map_new['namespaces']['pred'] = "http://www.semanticweb.org/esa-ioa/ontologies/2022/predicates-ontology#"
    data_map_new['baseIRI'] = "http://ioa-graph/resource/"
    return data_map_new
    