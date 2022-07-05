#./t5_eval.sh -f -d obqa -t unifiedqa-large -- ../../SBERT_OBQA/datasets/OpenBookQA/test/complete.jsonl
#./t5_eval.sh -f -d sciq -t unifiedqa-large -m unifiedqa-large_sciq_4e -- ../../../unifiedqa_datasets/sciq/test.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-v2-large-test -- ../../../unifiedqa_datasets/sciq/dev.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-v2-large-test -- ../../../unifiedqa_datasets/sciq/test.jsonl
