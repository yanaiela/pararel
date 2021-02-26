import argparse

import numpy as np
import pandas as pd

keys_dict = {
    'paraphrase': 1,
    'distractor': 2
}


def collect_answers(row):
    answers = []
    distractors = []
    for i in range(1, 8):
        rel_para_decision = keys_dict[row[f'Input.answer_{i}']]
        human_ans = row[f'Answer.ans{i}']

        if rel_para_decision == 1:
            answers.append(human_ans)
        else:
            distractors.append(human_ans)
    return answers, distractors


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-answers", "--answers", type=str, help="path to answers file",
                       default="crowdsourcing/answer_pilot.csv")

    args = parse.parse_args()

    np.random.seed(16)

    df = pd.read_csv(args.answers)

    all_answers, distractors = [], []
    total_paraphrases, total_distractors = 0, 0
    for _, row in df.iterrows():
        ans, dist = collect_answers(row)
        all_answers.extend(ans)
        distractors.extend(dist)
        total_paraphrases += len(ans)
        total_distractors += len(dist)

    print('para quality:', all_answers.count(1) / len(all_answers))
    print('distractors quality', distractors.count(2) / len(distractors))

    print('total paraphrases:', total_paraphrases)
    print('total distractors:', total_distractors)


if __name__ == '__main__':
    main()
