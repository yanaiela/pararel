cd /PATH-TO-DIR/
export PYTHONPATH=/PATH-TO-DIR

data_file=$1
lm=$2
graph=$3


/PATH-TO-ENV/bin/python pararel/consistency/encode_consistency_probe.py \
        --data_file $data_file \
        --lm $lm \
        --graph $graph \
        --gpu 0 \
        --wandb \
        --use_targets

