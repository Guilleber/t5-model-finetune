#!/bin/bash

show_help() {
cat << EOF
Usage: ${0##*/} [-hfc] [-d DATASET_ID] [-t MODEL_TYPE] [-m CHECKPOINT] [FILE]...
Translates a mcqa jsonl file using the specified model and returns the accuracy of the model on this task

    -h             display this help and exit
    -f             force rebuild of all intermediary files
    -c             clear temporary files after the script
    -d DATASET_ID  used when naming intermediary files          
    -t MODEL_TYPE  define the hyperparameters that are used. Model types are defined in ../settings.py
    -m CHECKPOINT  define the checkpoint from which the model will be loaded. If left unspecified, a new
                   model is created using MODEL_TYPE.
EOF
}

convert() {
if [ ! -e $input_tsv ] || [ $force_rebuild == 1 ]
then
    python3 obqa2seq.py $obqa_file $input_tsv
else
    echo "***Reusing file $input_tsv"
fi
}

generate() {
if [ ! -e $output_tsv ] || [ $force_rebuild == 1 ]
then
    if [ -z $checkpoint_name ]
    then
        python3 generate.py $input_tsv $output_tsv --model-type $model_type
    else
        python3 generate.py $input_tsv $output_tsv --model-type $model_type --checkpoint "../saved_models/$checkpoint_name.ckpt"
    fi
else
    echo "***Reusing file $output_tsv"
fi
}

prediction() {
if [ ! -e $pred_file ] || [ $force_rebuild == 1 ]
then
    python3 compute_pred.py $output_tsv $obqa_file $pred_file --model-type $model_type
else
    echo "***Reusing file $pred_file"
fi
}

accuracy() {
python3 compute_acc.py $obqa_file $pred_file
}


model_type=unifiedqa-large
dataset_id=defdata
force_rebuild=0
clear_files=0


while getopts hfct:m:d: opt; do
    case $opt in
        h)
	    show_help
	    exit 0
	    ;;
	f)
            force_rebuild=1
	    ;;
	c)
	    clear_files=1
	    ;;
	t)
	    model_type=$OPTARG
	    ;;
	m)
	    checkpoint_name=$OPTARG
	    ;;
	d)
	    dataset_id=$OPTARG
	    ;;
	*)
	    show_help >&2
	    exit 1
	    ;;
    esac
done
shift "$((OPTIND-1))"

obqa_file=$1
input_tsv="./temp/input_${dataset_id}.tsv"
output_tsv="./temp/output_${dataset_id}_${model_type}_${checkpoint_name:-new}.tsv"
pred_file="./temp/pred_${dataset_id}_${model_type}_${checkpoint_name:-new}.txt"

convert
generate
prediction
accuracy

if [ clear_files == 1 ]
then
    rm $input_tsv
    rm $output_tsv
    rm $pred_file
fi
