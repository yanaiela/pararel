from copy import deepcopy
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import pipeline, Pipeline, BertForMaskedLM, BertTokenizer


def parse_prompt(prompt: str, subject_label: str, object_label: str) -> str:
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)\
                   .replace(OBJ_SYMBOL, object_label)
    return prompt


# get mlm model to predict masked token.
def build_model_by_name(lm: str, args) -> Pipeline:
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """

    device = args.gpu
    if not torch.cuda.is_available():
        device = -1

    if 'consistancy' in lm:
        model = BertForMaskedLM.from_pretrained(lm)
        tokenizer = BertTokenizer.from_pretrained("bert-large-cased-whole-word-masking")
        model = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device, top_k=100)
    else:
        model = pipeline("fill-mask", model=lm, device=device, top_k=100)

    return model


def tokenize_results(results, pipeline_model, possible_objects):
    if pipeline_model.model.config.model_type in ['roberta']:
        preds_tokenized = []
        for example in results:
            example_tokenized = []
            for ans in example:
                ans_copy = deepcopy(ans)
                original_obj_ans = pipeline_model.tokenizer.convert_tokens_to_string(ans['token_str'])
                ans_copy['token_str'] = original_obj_ans

                example_tokenized.append(ans_copy)
            preds_tokenized.append(example_tokenized)
        return preds_tokenized
    else:
        return results


def run_query(pipeline_model: Pipeline, vals_dic: List[Dict], prompt: str, possible_objects: List[str], bs: int = 20)\
        -> (List[Dict], List[Dict]):
    data = []

    mask_token = pipeline_model.tokenizer.mask_token

    # create the text prompt
    for sample in vals_dic:
        data.append({'prompt': parse_prompt(prompt, sample["sub_label"], mask_token),
                     'sub_label': sample["sub_label"], 'obj_label': sample["obj_label"]})

    batched_data = []
    for i in range(0, len(data), bs):
        batched_data.append(data[i: i + bs])

    predictions = []
    for batch in tqdm(batched_data):
        preds = pipeline_model([sample["prompt"] for sample in batch], targets=possible_objects)
        # pipeline_model returns a list in case there is only 1 item to predict (in contrast to list of lists)
        if len(batch) == 1:
            preds = [preds]
        tokenized_preds = tokenize_results(preds, pipeline_model, possible_objects)
        predictions.extend(tokenized_preds)

    data_reduced = []
    for row in data:
        if pipeline_model.model.config.model_type in ['albert']:
            data_reduced.append({'sub_label': row['sub_label'],
                                 'obj_label': pipeline_model.tokenizer.tokenize(row['obj_label'])[0]})
        elif pipeline_model.model.config.model_type in ['roberta']:
            data_reduced.append({'sub_label': row['sub_label'],
                                 'obj_label': ' ' + row['obj_label']})
        else:
            data_reduced.append({'sub_label': row['sub_label'], 'obj_label': row['obj_label']})

    preds_reduced = []
    for top_k in predictions:
        vals = []
        for row in top_k:
            vals.append({'score': row['score'], 'token': row['token'], 'token_str': row['token_str']})
        preds_reduced.append(vals)

    return data_reduced, preds_reduced

