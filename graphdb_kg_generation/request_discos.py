"""Helper functions for requesting from DISCOSWeb"""

import requests
from pprint import pprint
from tqdm.notebook import trange
import time
import pdb

URL = 'https://discosweb.esoc.esa.int'
with open('token-discos.txt') as f: #put your own discos token in this file
    token = f.read()
    
    
def countdownTimer(Nsec, message=''):
    t = trange(Nsec, leave=False)
    t.set_description(message + f'Wait {Nsec} seconds')
    for _ in t:
        time.sleep(1)
        

def requestSinglePage(pageName="/api/objects",pageSize=20,filter=None,sort=None,include=None):
    """Return JSON format of response from a single page"""
    with requests.Session() as s:
        response = s.get(
            URL + pageName,
            headers={
                'Authorization': f'Bearer {token}',
                'DiscosWeb-Api-Version': '2',
            },
            params={
                'page[size]': pageSize,
                'filter': filter,
                'sort' : sort,
                'include' : include,
            },
        )
        resp_json = response.json()
        if response.ok:
            return resp_json
        else:
            pprint(resp_json['errors'])
            if resp_json['errors'][0]['status']=='429':
                print(f"Retry after: {response.headers['Retry-After']}")
            return []

def requestAllPages(pageName="/api/objects",pageSize=100,filter=None,sort=None,include=None):
    """Iterate over all pages and return data from JSON format of response"""
    with requests.Session() as s:
        response = s.get(
            URL + pageName,
            headers={
                'Authorization': f'Bearer {token}',
                'DiscosWeb-Api-Version': '2',
            },
            params={
                'page[size]': pageSize,
                'filter': filter,
                'sort' : sort,
                'include' : include,
            },
        )
        resp_json = response.json()
        if not response.ok:
            pprint(resp_json['errors'])
            if resp_json['errors'][0]['status']=='429':
                print(f"Retry after: {response.headers['Retry-After']}")
            return []

        allData = resp_json['data']
        nextLink = resp_json['links']['next']

        while nextLink is not None:
            response = s.get(
                URL + nextLink,
                headers={
                    'Authorization': f'Bearer {token}',
                    'DiscosWeb-Api-Version': '2',
                },
            )
            resp_json = response.json()
            if not response.ok:
                pprint(resp_json['errors'])
                if resp_json['errors'][0]['status']=='429':
                    print(f"Retry after: {response.headers['Retry-After']}")
                return allData

            allData.extend(resp_json['data'])
            nextLink = resp_json['links']['next']

    return allData

def requestWithRelationships(pageName="/api/objects",pageSize=100,filter=None,sort=None,include=None):
    """All pages, also return data from 'included'"""
    with requests.Session() as s:
        response = s.get(
            URL + pageName,
            headers={
                'Authorization': f'Bearer {token}',
                'DiscosWeb-Api-Version': '2',
            },
            params={
                'page[size]': pageSize,
                'filter': filter,
                'sort' : sort,
                'include' : include,
            },
        )
        resp_json = response.json()
        if not response.ok:
            pprint(resp_json['errors'])
            if resp_json['errors'][0]['status']=='429':
                print(f"Retry after: {response.headers['Retry-After']}")
            return []

        allData = resp_json['data']
        try:
            incData = resp_json['included']
        except KeyError:
            incData = []
        nextLink = resp_json['links']['next']

        while nextLink is not None:
            response = s.get(
                URL + nextLink,
                headers={
                    'Authorization': f'Bearer {token}',
                    'DiscosWeb-Api-Version': '2',
                },
            )
            resp_json = response.json()
            if not response.ok:
                pprint(resp_json['errors'])
                if resp_json['errors'][0]['status']=='429':
                    print(f"Retry after: {response.headers['Retry-After']}")
                return allData

            allData.extend(resp_json['data'])
            try:
                incData.extend(resp_json['included'])
            except KeyError:
                pass
            nextLink = resp_json['links']['next']

    return allData, incData


def requestAllAndWait(pageName="/api/objects",pageSize=100,filter=None,sort=None,include=None):
    """For larger requests, waits for required time upon a 429 error"""
    with requests.Session() as s:
        response = s.get(
            URL + pageName,
            headers={
                'Authorization': f'Bearer {token}',
                'DiscosWeb-Api-Version': '2',
            },
            params={
                'page[size]': pageSize,
                'filter': filter,
                'sort' : sort,
                'include' : include,
            },
        )
        resp_json = response.json()
        if not response.ok:
            pprint(resp_json['errors'])
            if resp_json['errors'][0]['status']=='429':
                print(f"Retry after: {response.headers['Retry-After']}")
            return ([],[])

        allData = resp_json['data']
        try:
            incData = resp_json['included']
        except KeyError:
            incData = []
        nextLink = resp_json['links']['next']
        totalPages = resp_json['meta']['pagination']['totalPages']

        while nextLink is not None:
            response = s.get(
                URL + nextLink,
                headers={
                    'Authorization': f'Bearer {token}',
                    'DiscosWeb-Api-Version': '2',
                },
            )
            resp_json = response.json()
            if not response.ok:
                if resp_json['errors'][0]['status']=='429':
                    message = f'Page {currentPage}/{totalPages}. '
                    t_wait = int(response.headers['Retry-After'])+1
                    countdownTimer(t_wait, message=message)
                    continue
                else:
                    pprint(resp_json['errors'])
                    return allData, incData

            currentPage = resp_json['meta']['pagination']['currentPage']
            allData.extend(resp_json['data'])
            try:
                incData.extend(resp_json['included'])
            except KeyError:
                pass
            nextLink = resp_json['links']['next']

    return allData, incData
