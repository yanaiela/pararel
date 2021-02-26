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
            'albert-xxlarge-v2',

            'nyu-mll/roberta-base-1B-1',
            'nyu-mll/roberta-base-100M-1',
            'nyu-mll/roberta-base-10M-1',
            'nyu-mll/roberta-med-small-1M-1',

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
        if relation_id != 'P106': continue
        for encoder in encoders:

            prediction_encoder_name = encoder
            if 'nyu-mll' in prediction_encoder_name:
                prediction_encoder_name = encoder.split('/')[-1]
            cartesian_product.append([f'data/pattern_data/graphs_json/{relation_id}.jsonl',
                                      f'data/trex_lms_vocab/{relation_id}.jsonl',
                                      encoder,
                                      f'data/output/representations/{relation_id}_{prediction_encoder_name}.npy'])

    parallelize(nodes, cartesian_product, '/PATH-TO-DIR/pararel/runs/pararel/encode_text.sh',
                on_gpu=False, dry_run=args.dry_run)
