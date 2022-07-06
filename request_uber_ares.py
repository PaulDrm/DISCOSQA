"""Helper functions for requesting from Uberlog and ARES"""

import requests
from pprint import pprint
import pdb

URL_ares = 'http://localhost:8081/ioa/adapter/ares/'
URL_uberlog = 'http://localhost:8081/ioa/adapter/uberlog/'

def requestAres(mission, parameter, intervalStart, intervalEnd, 
                pageName="data", operation=None):
    params = {'mission':mission,
              'parameter':parameter,
              'intervalStart':intervalStart,
              'intervalEnd':intervalEnd}
    if pageName=="aggregate" and operation is not None:
        params['operation'] = operation
    with requests.Session() as s:
        response = s.get(
            URL_ares + pageName,
            params=params
        )

        resp_json = response.json()
        if response.ok:
            return resp_json
        else:
            pprint(resp_json['detail'])
            return []
        
def requestUberlog(mission, pageName="entries", **kwargs):
    """
    Optional keyword arguments
    __________________________
    
    eventDateFrom : int
    eventDateTo : int
    author : string
    severity : string
    text : integer - needs confirmed
    sort : bool
    skip : integer
    limit : integer
    """
    with requests.Session() as s:
        response = s.get(
            URL_uberlog + pageName,
            params = kwargs
        )

        resp_json = response.json()
        if response.ok:
            return resp_json
        else:
            pprint(resp_json['detail'])
            return []
