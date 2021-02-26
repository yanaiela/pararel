import json


def get_lama_patterns(in_file):
    with open(in_file, 'r') as f:
        relations = f.readlines()
        relations = [json.loads(x.strip())['relation'] for x in relations]

    # these relations are not in LAMA's data
    relations = [x for x in relations if x not in ['P166', 'P69', 'P54', 'P1923', 'P102'] +
                 # Relations that are hard to create patterns for
                 # or their subjects are mixed
                 ['P527', 'P31']]

    return relations
