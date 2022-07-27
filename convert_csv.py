"""Functions for converting csv files to RDF triples"""

import numpy as np
from pprint import pprint
from tqdm.notebook import trange
import pandas as pd
import pdb


def row2rdf(dfRow):
    """Converts data from row of DataFrame to rdf triples"""
    rdf_str = f"<{dfRow['name']}> a ioa:{dfRow['class']} ;\n"
    rdf_str += f"  ioa:hasName '{dfRow['name']}' ;\n"
    if dfRow['parent'] is not np.nan:
        for parent in dfRow['parent']:
            rdf_str += f"  ioa:hasParent <{parent}> ;\n"
    rdf_str += f"  rdfs:label '{dfRow['label']}' ."
    return rdf_str


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
    