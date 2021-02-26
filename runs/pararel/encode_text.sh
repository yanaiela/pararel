cd /PATH-TO-DIR/
export PYTHONPATH=/PATH-TO-DIR

patterns_file=$1
data_file=$2
lm=$3
pred_file=$4


/PATH-TO-ENV/bin/python pararel/encode/encode_text.py \
        --patterns_file $patterns_file \
        --data_file $data_file \
        --lm $lm \
        --pred_file $pred_file \
        --wandb

