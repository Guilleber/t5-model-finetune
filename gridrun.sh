#python3 t5_train.py --model-name t5-large --datasets openbookqa --epochs 4 --gpus 1 --save-model ./stored_models/baseline_t5-large_obqa.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_wiki_bis --epochs 2 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_wiki_bis_2e.ckpt
/usr/bin/python3 train.py --model_name unifiedqa-large-lr1 --datasets qasc --epochs 4 --gpus 1 --save_best_model --exp_name qasc
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets enhanced_synthetic_wiki --epochs 4 --gpus 1 --save-model ./stored_models/unifiedqa-large_enhanced_synthetic_wiki.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_sciq --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_sciq.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_wiki --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_wiki.ckpt
#python3 t5_train.py --model-name unifiedqa-large-lr1 --datasets synthetic_obqa --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_obqa.ckpt
#python3 t5_train.py --model-name t5-large --datasets gigaword --epochs 1 --gpus 1 --save-model ./stored_models/t5-large_summarization.ckpt
