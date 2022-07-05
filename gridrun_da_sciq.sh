#python3 t5_train.py --model-name t5-large --datasets openbookqa --epochs 4 --gpus 1 --save-model ./stored_models/baseline_t5-large_obqa.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_wiki_bis --epochs 2 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_wiki_bis_2e.ckpt
/usr/bin/python3 train.py --model_name unifiedqa-large-lr1 --datasets "!enhanced_synthetic_wiki_2 100:sciq" --epochs 8 --gpus 1 --save_best_model --exp_name unifiedqa-large_da_esw_sciq_100
/usr/bin/python3 train.py --model_name unifiedqa-large-lr1 --datasets "!enhanced_synthetic_wiki_2 75:sciq" --epochs 12 --gpus 1 --save_best_model --exp_name unifiedqa-large_da_esw_sciq_75
/usr/bin/python3 train.py --model_name unifiedqa-large-lr1 --datasets "!enhanced_synthetic_wiki_2 50:sciq" --epochs 16 --gpus 1 --save_best_model --exp_name unifiedqa-large_da_esw_sciq_50
/usr/bin/python3 train.py --model_name unifiedqa-large-lr1 --datasets "!enhanced_synthetic_wiki_2 25:sciq" --epochs 32 --gpus 1 --save_best_model --exp_name unifiedqa-large_da_esw_sciq_25
#/usr/bin/python3 train.py --model_name unifiedqa-large-lr1 --datasets "100:sciq" --epochs 8 --gpus 1 --save_best_model --exp_name unifiedqa-large_sciq_100
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets enhanced_synthetic_wiki --epochs 4 --gpus 1 --save-model ./stored_models/unifiedqa-large_enhanced_synthetic_wiki.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_sciq --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_sciq.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_wiki --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_wiki.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_obqa --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_obqa.ckpt
#python3 t5_train.py --model-name t5-large --datasets gigaword --epochs 1 --gpus 1 --save-model ./stored_models/t5-large_summarization.ckpt
