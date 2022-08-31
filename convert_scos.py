"""Functions for converting S2k files to RDF triples"""

import numpy as np
from pprint import pprint
from tqdm.notebook import trange
import pandas as pd
import pdb


def row2rdf_reportingdata(dfRow):
    """Converts data from row of reporting data DataFrame to rdf triples"""
    rdf_str = f"<PCF_{dfRow['PCF_NAME']}> a ioa:ReportingData ;\n"
    rdf_str += f"  ioa:Name '{dfRow['PCF_NAME']}'^^xsd:string ;\n"
    rdf_str += f"  ioa:Description '{dfRow['PCF_DESCR']}'^^xsd:string ;\n"
    
    if not np.isnan(dfRow['PCF_PID']):
        rdf_str += f"  ioa:PID '{int(dfRow['PCF_PID'])}'^^xsd:unsignedInt ;\n"
        
    if type(dfRow['PCF_UNIT'])==str:
        rdf_str += f"  ioa:Unit '{dfRow['PCF_UNIT']}'^^xsd:string ;\n"
    
    rdf_str += f"  pred:name '{dfRow['PCF_NAME']}'^^xsd:string ;\n"
    rdf_str += f"  ioa:hasParent <{dfRow['parent_name']}> ."
    
    return rdf_str
    