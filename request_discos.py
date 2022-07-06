"""Helper functions for requesting from DISCOSWeb"""

import requests

URL = 'https://discosweb.esoc.esa.int'
with open('token-discos.txt') as f:
    token = f.read()

def requestSinglePage(pageName="/api/objects",pageSize=20,filter=None,sort=None,include=None):
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