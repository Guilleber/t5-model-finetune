#./t5_eval.sh -f -d obqa -t unifiedqa-large -- ../../SBERT_OBQA/datasets/OpenBookQA/test/complete.jsonl
#./t5_eval.sh -f -d sciq -t unifiedqa-large -m unifiedqa-large_sciq_4e -- ../../../unifiedqa_datasets/sciq/test.jsonl
data=da_esw_sciq
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_25" -- ../../../unifiedqa_datasets/sciq/dev.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_50" -- ../../../unifiedqa_datasets/sciq/dev.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_75" -- ../../../unifiedqa_datasets/sciq/dev.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_100" -- ../../../unifiedqa_datasets/sciq/dev.jsonl
#./t5_eval.sh -f -d sciq -t unifiedqa-large -m unifiedqa-large_enhanced_synthetic_wiki_3e -- ../../../unifiedqa_datasets/sciq/test.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_25" -- ../../../unifiedqa_datasets/sciq/test.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_50" -- ../../../unifiedqa_datasets/sciq/test.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_75" -- ../../../unifiedqa_datasets/sciq/test.jsonl
./t5_eval.sh -f -d sciq -t unifiedqa-large -m "unifiedqa-large_${data}_100" -- ../../../unifiedqa_datasets/sciq/test.jsonl
#./t5_eval.sh -f -d sciq -t unifiedqa-large -m unifiedqa-large_enhanced_synthetic_wiki_3e -- ../../../unifiedqa_datasets/sciq/dev.jsonl
#./t5_eval.sh -f -d sciq -t unifiedqa-large -m unifiedqa-large_enhanced_synthetic_sciq_64 -- ../../../unifiedqa_datasets/sciq/test.jsonl
#./t5_eval.sh -f -d sciq -t unifiedqa-large -- ../../../unifiedqa_datasets/sciq/dev.jsonl
