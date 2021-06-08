# ParaRel :metal:

This repository contains the code and data for the paper:

[`Measuring and Improving Consistency in Pretrained Language Models`](https://arxiv.org/abs/2102.01017)

as well as the resource: `ParaRel` :metal:


Since this work required running a lot of experiments, it is structured by scripts that automatically 
runs many sub-experiments, on parallel servers, and tracking using an experiment tracking website: [wandb](https://wandb.ai/site),
which are then aggregated using a jupyter notebook.
To run all the experiments I used [task spooler](https://vicerveza.homeunix.net/~viric/soft/ts/), a queue-based software
that allows to run multiple commands in parallel (and store the rest in a queue)

It is also possible to run individual experiments, for which one can look for in the corresponding script.

For any question, query regarding the code, or paper, please reach out at `yanaiela@gmail.com`


## ParaRel :metal:
If you're only interested in the data, you can find it under [data](data/pattern_data/graphs_json).
Each file contains the paraphrases patterns for a specific relation, in a json file.



## Create environment
```sh
conda create -n pararel python=3.7 anaconda
conda activate pararel

pip install -r requirements.txt
```
add project to path:
```sh
export PYTHONPATH=${PYTHONPATH}:/path-to-project
```


## Setup

In case you just want to start with the filtered data we used (filtering objects that consist more than a single word
piece in the LMs we considered), you can find them [here](data/trex_lms_vocab/).
Otherwise:

First, begin by downloading the trex dataset from [here](https://dl.fbaipublicfiles.com/LAMA/data.zip),
alternatively, check out the [LAMA github repo](https://github.com/facebookresearch/LAMA).
Download it to the following [folder](data/trex/) so that the following folder would exist:
`data/trex/data/TREx` along with the relevant files


Next, in case you want to rerun automatically some/all of the experiments, you will need to update 
the paths in the [runs](runs/) scripts with your folder path and virtual environment.

## Run Scripts

Filter data from trex, to include only triplets that appear in the inspected LMs in this work:
`bert-base-cased`, `roberta-base`, `albert-base-v2` (as well as the larger versions, that contain the same vocabulary)
```sh
python runs/pararel/filter.py
```

A single run looks like the following:
```sh
python lm_meaning/lm_entail/filter_data.py \
       --in_data data/trex/data/TREx/P106.jsonl \
       --model_names bert-base-cased,bert-large-cased,bert-large-cased-whole-word-masking,roberta-base,roberta-large,albert-base-v2,albert-xxlarge-v2 \
       --out_file data/trex_lms_vocab/P106.jsonl
```

Evaluate consistency:
```sh
python runs/eval/run_lm_consistent.py
```

A single run looks like the following:
```sh
python pararel/consistency/encode_consistency_probe.py \
       --data_file data/trex_lms_vocab/P106.jsonl \
       --lm bert-base-cased \
       --graph data/pattern_data/graphs/P106.graph \
       --gpu 0 \
       --wandb \
       --use_targets
```

Encode the patterns along with the subjects, to save the representations:
```sh
python runs/pararel/encode_text.py
```

A single run looks like the following:
```sh
python lm_meaning/encode/encode_text.py \
       --patterns_file data/pattern_data/graphs_json/P106.jsonl \
       --data_file data/trex_lms_vocab/P106.jsonl \
       --lm bert-base-cased \
       --pred_file data/output/representations/P106_bert-base-cased.npy \
       --wandb
```

## Improving Consistency with ParaRel
The code and README are available [here](pararel/ft)

## FAQ

Q: Why do you report 31 N-1 relations, whereas in the LAMA paper there are only 25?

A: [Explanation](https://github.com/yanaiela/pararel/wiki/31-N1-Relations)

## Citation:
If you find this work relevant to yours, please cite us:
```
@article{Elazar2021MeasuringAI,
  title={Measuring and Improving Consistency in Pretrained Language Models},
  author={Yanai Elazar and Nora Kassner and Shauli Ravfogel and Abhilasha Ravichander and Ed Hovy and Hinrich Schutze and Yoav Goldberg},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.01017}
}
```
