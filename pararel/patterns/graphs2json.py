import argparse
import pickle
from glob import glob

from tqdm import tqdm

from pararel.consistency.utils import write_jsonl_file


def get_patterns(in_f):
    with open(in_f, 'rb') as f:
        data = pickle.load(f)

    patterns = []
    for node in list(data.nodes):
        patterns.append({
            'pattern': node.lm_pattern,
            'lemma': node.lemma,
            'extended_lemma': node.extended_lemma,
            'tense': node.tense
        })
    return patterns


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--graphs_path", type=str, help="Path to relations graphs", default="data/pattern_data/graphs/")
    parse.add_argument("--out_path", type=str, help="Output dir path to save json files of the patterns",
                       default="data/pattern_data/graphs_json/")

    args = parse.parse_args()

    for f in tqdm(glob(args.graphs_path + '/*.graph')):
        rel_name = f.split('/')[-1].split('.')[0]
        print(rel_name)
        if rel_name in ['P527', 'P31']: continue
        json_patterns = get_patterns(f)

        write_jsonl_file(json_patterns, args.out_path + '/' + rel_name + '.jsonl')


if __name__ == '__main__':
    main()
