#./t5_eval.sh -f -d qasc -t unifiedqa-large -- ../../../unifiedqa_datasets/qasc/dev.jsonl
./t5_eval.sh -f -d csqa -t unifiedqa-large -m unifiedqa-large_cqa -- ../../../unifiedqa_datasets/commonsenseqa/dev.jsonl
