import json

def get_easy():

    json_loader = open('data/easy.txt', 'rb')
    json_loader_attack = open('data/easy.attack.txt', 'rb')

    attributes = ['method', 'url', 'name', 'age', 'origin']


    data_attributes_attack = {}
    data_attributes = {}
    for a in attributes:
        data_attributes[a] = []
        data_attributes_attack[a] = []

    data_attributes['all'] = []
    data_attributes_attack['all'] = []

    for json_string in json_loader:
        data_attributes['all'].append(json_string.rstrip().decode())
        json_request = json.loads(json_string.rstrip().decode())
        for a in attributes:
            data_attributes[a].append(str(json_request[a]))

    for json_string in json_loader_attack:
        data_attributes_attack['all'].append(json_string.rstrip().decode())
        json_request = json.loads(json_string.rstrip().decode())
        for a in attributes:
            data_attributes_attack[a].append(str(json_request[a]))
            
    return data_attributes, data_attributes_attack
