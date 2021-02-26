cd /PATH-TO-DIR/
export PYTHONPATH=/PATH-TO-DIR

patterns_file=$1
model_names=$2
out_file=$3


/PATH-TO-ENV/bin/python pararel/patterns/filter_data.py \
        --in_data $patterns_file \
        --model_names $model_names \
        --out_file $out_file
