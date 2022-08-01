"""Functions for converting csv files to RDF triples"""

import numpy as np
from pprint import pprint
from tqdm.notebook import trange
import pandas as pd
import pdb


def row2rdf(dfRow):
    """Converts data from row of DataFrame to rdf triples"""
    itemClass = dfRow['class']
    att_str = dfRow['attributes']
    if att_str is not np.nan:
        type_map = {str:'string', int:'unsignedInt', float:'float'}
        att_dict = dict( (k.strip(), strOrNum(v)) for k,v in
                       (s.split(':') for s in att_str.split(';')) )
    
    if itemClass != 'Spacecraft':
        rdf_str = f"<{dfRow['name']}> a ioa:{dfRow['class']} ;\n"
        rdf_str += f"  ioa:Name '{dfRow['name']}'^^xsd:string ;\n"
    else:
        rdf_str = f"<obj{att_dict['DiscosID']}> a ioa:{dfRow['class']} ;\n"
        
    if att_str is not np.nan:
        for att,val in att_dict.items():
            val_type = type(val)
            rdf_str += f"  ioa:{att} '{val}'^^xsd:{type_map[val_type]} ;\n"
    
    if dfRow['parent'] is not np.nan:
        for parent in dfRow['parent']:
            rdf_str += f"  ioa:hasParent <{parent}> ;\n"
    rdf_str += f"  rdfs:label '{dfRow['label']}'^^xsd:string ."
    return rdf_str


def strOrNum(s):
    """Returns string if s contains ', else returns int or float'"""
    s = s.strip()
    if "'" in s:
        return s.strip("'")
    elif "." in s:
        return float(s)
    else:
        return int(s)


def renameDF(dataDF):
    """Renames 'parent', 'name', and 'label' columns according to the mission"""
    
    missionClass = dataDF['class']=='Mission'
    missionName = dataDF.name.loc[missionClass].values[0]
    missionLabel = dataDF.label.loc[missionClass].values[0]
    parentMission = dataDF['parent']==missionName
    
    dataDF['name'] = dataDF.name.map(lambda s: s+'_mission' if s==missionName else s)
    dataDF['parent'] = dataDF.parent.map(lambda s: s+'_mission' if s==missionName else s)
    dataDF.name[~missionClass] = dataDF.name[~missionClass].map(lambda s: missionName+'_'+s)
    dataDF.label[~missionClass] = dataDF.label[~missionClass].map(lambda s: missionLabel+' '+s)
    
    dataDF.parent[~missionClass] = dataDF.parent[~missionClass].map(lambda s: s.split(";"))
    parentMask = ~missionClass&~parentMission
    dataDF.parent[parentMask] = dataDF.parent[parentMask].map(lambda s_list: [missionName + '_' + s for s in s_list])
    
    return dataDF
    