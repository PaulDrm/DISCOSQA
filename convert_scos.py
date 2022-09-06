"""Functions for converting S2k files to RDF triples"""

import numpy as np
from pprint import pprint
from tqdm.notebook import trange
import pandas as pd
import pdb


def row2rdf_reportingdata(dfRow, missionName):
    """Converts data from row of reporting data DataFrame to rdf triples"""
    
    rdf_str = f"<PCF_{dfRow['PCF_NAME']}> a ioa:ReportingData ;\n"
    rdf_str += f"  pred:instance_of ioa:ReportingData ;\n"
    rdf_str += f"  ioa:Name '{dfRow['PCF_NAME']}'^^xsd:string ;\n"
    rdf_str += f"  ioa:Description '{dfRow['PCF_DESCR']}'^^xsd:string ;\n"
    
    if not np.isnan(dfRow['PCF_PID']):
        rdf_str += f"  ioa:PID '{int(dfRow['PCF_PID'])}'^^xsd:unsignedInt ;\n"
        
    if type(dfRow['PCF_UNIT'])==str:
        rdf_str += f"  ioa:Unit '{dfRow['PCF_UNIT']}'^^xsd:string ;\n"
    
    #API call string should be edited
    rdf_str += f"  ioa:APICall '/ares/data?mission={missionName}&parameter={dfRow['PCF_NAME']}&intervalStart={{}}&intervalEnd={{}}'^^xsd:string ;\n"
    rdf_str += f"  pred:name '{dfRow['PCF_NAME']}'^^xsd:string ;\n"
    rdf_str += f"  ioa:parent <{dfRow['parent_name']}> ."
    
    return rdf_str

def row2rdf_activity(dfRow, missionName):
    """Converts data from row of activity DataFrame to rdf triples"""
    
    rdf_str = f"<CCF_{dfRow['CCF_CNAME']}> a ioa:Activity ;\n"
    rdf_str += f"  pred:instance_of ioa:Activity ;\n"
    rdf_str += f"  ioa:Name '{dfRow['CCF_CNAME']}'^^xsd:string ;\n"
    rdf_str += f"  ioa:Description '{dfRow['CCF_DESCR']}'^^xsd:string ;\n"
    
    if type(dfRow['CCF_DESCR2'])==str:
        rdf_str += f"  ioa:DetailedDescription '{dfRow['CCF_DESCR2']}'^^xsd:string ;\n"
    
    rdf_str += f"  pred:name '{dfRow['CCF_CNAME']}'^^xsd:string ;\n"
    rdf_str += f"  ioa:parent <{dfRow['parent_name']}> ."
    
    return rdf_str

def row2rdf_event(dfRow, missionName):
    """Converts data from row of event DataFrame to rdf triples"""
    
    rdf_str = f"<PID_{dfRow.name}> a ioa:Event ;\n"
    rdf_str += f"  pred:instance_of ioa:Event ;\n"
    rdf_str += f"  ioa:Description '{dfRow['PID_DESCR']}'^^xsd:string ;\n"

    rdf_str += f"  ioa:APICall '/uberlog/entries?mission={missionName}&eventDateFrom={{}}&eventDateTo={{}}&text={dfRow['PID_DESCR'].replace(' ','%20')}'^^xsd:string ;\n"
    rdf_str += f"  ioa:parent <{dfRow['parent_name']}> ."
    
    return rdf_str