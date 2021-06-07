# Finetune consistency

## Training

To finetune the model on a provided sample dataset run the following command:

```sh
python pararel/ft/train_consistency.py --dataset_name pararel/ft/data/100_3_P37-P138-P449/ --mlm_LAMA pararel/ft/data/100_3_P37-P138-P449/train_mlm.txt --candidate_set
```

## Generate data

In case you want to use different relations, subject-object tuples or a
different number if relations, you can use this script to generate your own
training data.

```sh
python pararel/ft/generate_data_ft_consitency.py
```
with optional parameters:

-`num_relations`: how man relations should be used (default 3) \
-`num_tuples`: how many subject-object tuples should be used (default 100) \
-`relations_given`: which relations should be used (default "P138,P449,P37")
