#./t5_eval.sh -f -d qasc -t unifiedqa-large -- ../../../unifiedqa_datasets/qasc/dev.jsonl
./t5_eval.sh -f -d qasc -t unifiedqa-large -m unifiedqa-large_synthetic_wiki-v2 -- ../../../unifiedqa_datasets/qasc/test.jsonl
