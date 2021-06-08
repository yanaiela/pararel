# Finetune consistency

## Training

To finetune the model on a provided sample dataset run the following command:

```sh
python pararel/ft/train_consistency.py --dataset_name pararel/ft/data/100_3_P138_P37_P449/ --mlm_LAMA pararel/ft/data/100_3_P138_P37_P449/train_mlm.txt --candidate_set
```
-`dataset_name`: referring to the folder containing ft data

with optional parameters:

-`mlm_LAMA`:  provide path to a LAMA based corpus to alternate the consistency ft with mlm on LAMA (default "") \
-`num_LAMA_steps`: if mlm on LAMA is used set the number of mlm training steps on LAMA (default 5) \
-`mlm_wiki`: provide path to a wikipedia based corpus to alternate the consistency ft with mlm on wikipedia (default "")\
-`num_wiki_steps`:  if mlm on wikipedia is used set the number of mlm training steps on wikipedia (default 5) \
-`loss`: chose the ft loss (default "kl") \
with options: "kl" for KL-divergence, "cos"/"repcos" for cosine loss on the output/last hidden layer \
-`loss_scaling`: a parameter decreasing the consistency loss (default 0.8) \
-`candidate_set`: if this flag to computed loss using a candidate subset instead of the full vocabulary (default False)


## Generate data

In case you want to use different relations, subject-object tuples or a
different number of relations, you can use this script to generate your own
training data.

```sh
python pararel/ft/generate_data_ft_consistency.py
```
with optional parameters:

-`num_relations`: how man relations should be used (default 3) \
-`num_tuples`: how many subject-object tuples should be used (default 100) \
-`relations_given`: which relations should be used (default "P138,P449,P37")
