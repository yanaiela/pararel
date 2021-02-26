import argparse
from runs.ts_run import parallelize
from runs.utils import get_lama_patterns


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'server-name1',
    'server-name2',
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = ['bert-base-cased',
            'bert-large-cased',
            'bert-large-cased-whole-word-masking',
            'roberta-base',
            'roberta-large',
            'albert-base-v2',
            'albert-xxlarge-v2'
            ]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="data/trex/data/relations.jsonl")
    args = parse.parse_args()

    relations = get_lama_patterns(args.patterns)

    cartesian_product = []
    for relation_id in relations:
        cartesian_product.append([f'data/trex/data/TREx/{relation_id}.jsonl',
                                  ','.join(encoders),
                                  f'data/trex_lms_vocab/{relation_id}.jsonl'])

    parallelize(nodes, cartesian_product, '/PATH-TO-DIR/pararel/runs/pararel/filter.sh',
                on_gpu=False, dry_run=args.dry_run)
