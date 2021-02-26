import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
from glob import glob
import json
import numpy as np
import pandas as pd
from pararel.consistency.utils import read_graph


def sample_from_graph(graph_file, sample_size):
    patterns_graph = read_graph(graph_file)
    prompts = [x.lm_pattern for x in list(patterns_graph.nodes)]
    all_pairs = list(itertools.combinations(prompts, 2))
    if len(all_pairs) < sample_size:
        return []
    indices = np.random.choice(len(all_pairs), sample_size, replace=False)
    return np.array(all_pairs)[indices].tolist()


def get_patterns(graph_file, base_pattern):
    patterns_graph = read_graph(graph_file)
    prompts = [x.lm_pattern for x in list(patterns_graph.nodes)]
    no_base = []
    for p in prompts:
        if p.replace(' .', '.') == base_pattern.replace(' .', '.'):
            continue
        no_base.append(p)
    assert len(prompts) - 1 == len(no_base)
    combinations = list(itertools.product([base_pattern], no_base))
    return [list(x) for x in combinations]


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-graphs", "--graphs", type=str, help="folder where graphs are located",
                       default="data/pattern_data/graphs/")
    parse.add_argument("-trex", "--trex", type=str, help="folder where tuples are located",
                       default="data/trex_lms_vocab/")
    parse.add_argument("-relations", "--relations", type=str, help="path to relations file",
                       default="data/trex/data/relations.jsonl")
    parse.add_argument("-out_file", "--out_file", type=str, help="output csv file",
                       default="crowdsourcing/to_annotate.csv")
    parse.add_argument("--sample", type=int, default=10)
    parse.add_argument("--negative", type=int, default=2)

    args = parse.parse_args()

    np.random.seed(16)

    with open(args.relations, 'r') as f:
        rels = f.readlines()
        rels = [json.loads(x) for x in rels]
    rels2json = {x['relation']: x for x in rels}

    rel2tuples = {}
    for f in glob(args.trex + '/*.jsonl'):
        rel = f.split('/')[-1].split('.')[0]
        with open(f, 'r') as f:
            data = f.readlines()
            data = [json.loads(x) for x in data]
        tuples = [(x['sub_label'], x['obj_label']) for x in data]
        rel2tuples[rel] = tuples

    skipped = 0
    positive_pairs = {}
    for f in glob(args.graphs + '/*.graph'):
        rel = f.split('/')[-1].split('.')[0]
        if rel in ['P361', 'P106', 'P527', 'P31']: continue
        pairs = get_patterns(f, rels2json[rel]['template'])
        if len(pairs) == 0:
            skipped += 1
            continue

        pairs_output = []
        for x in pairs:
            nums = np.random.choice(len(rel2tuples[rel]), 1)[0]
            pairs_output.append(x + [rel, rel2tuples[rel][nums][0], rel2tuples[rel][nums][1], 'paraphrase'])

        positive_pairs[rel] = pairs_output

    print('num of skipped', skipped)
    all_relations = list(positive_pairs.keys())
    negative_pairs = defaultdict(list)
    for pos_rel in positive_pairs.keys():
        neg_rels = np.random.choice(all_relations, 5)
        for neg_rel in neg_rels:
            if pos_rel == neg_rel:
                continue
            if len(positive_pairs[neg_rel]) == 0 or len(positive_pairs[pos_rel]) == 0:
                continue
            row_pos = positive_pairs[pos_rel][np.random.choice(len(positive_pairs[pos_rel]), 1)[0]]
            row_neg = positive_pairs[neg_rel][np.random.choice(len(positive_pairs[neg_rel]), 1)[0]]
            neg_pair = deepcopy(row_pos)
            neg_pair[1] = row_neg[0]
            neg_pair[-1] = 'distractor'
            negative_pairs[pos_rel].append(neg_pair)

    output_data = []
    rest_data = []
    for rel in all_relations:
        l = positive_pairs[rel][:args.sample]
        if len(l) < args.sample:
            rest_data.extend(l)
            rest_data.extend(negative_pairs[rel][:args.negative - 1])
            continue
        l.extend(negative_pairs[rel][:args.negative])
        np.random.shuffle(l)
        output_data.append(l)

    np.random.shuffle(rest_data)
    for i in range(0, len(rest_data), 7):
        sub = rest_data[i: i + 7]
        if len(sub) < 7:
            continue
        output_data.append(sub)

    for_labeling = []
    for batch in output_data:
        if len(batch) == 0:
            continue
        fix_batch = [batch[0][2]]
        for row in batch:
            p1, p2 = row[:2]
            rel = row[2]
            subj, obj = row[3:5]
            ans = row[5]
            p1 = p1.replace('[X]', f'<i>{subj}</i>').replace('[Y]', f'<i>{obj}</i>')
            p2 = p2.replace('[X]', f'<i>{subj}</i>').replace('[Y]', f'<i>{obj}</i>')
            fix_batch.extend([p1, p2, ans])
        for_labeling.append(fix_batch)

    print(for_labeling[0])
    print(len(for_labeling))

    names = []
    for i in range(1, 1 + args.negative + args.sample):
        names.append(f'para_{i}_a')
        names.append(f'para_{i}_b')
        names.append(f'answer_{i}')
    df = pd.DataFrame(for_labeling, columns=['Pattern'] + names)
    df.to_csv(args.out_file, index=False)


if __name__ == '__main__':
    main()
