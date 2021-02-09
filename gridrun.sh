#python3 t5_train.py --model-name t5-large --datasets openbookqa --epochs 4 --gpus 1 --save-model ./stored_models/baseline_t5-large_obqa.ckpt
#python3 t5_train.py --model-name unifiedqa-large --datasets openbookqa --epochs 4 --gpus 1 --save-model ./stored_models/baseline_unifiedqa-large_obqa.ckpt
python3 t5_train.py --model-name unifiedqa-large --datasets synthetic --epochs 1 --gpus 1 --save-model ./stored_models/unifiedqa-large_synthetic_finetune.ckpt
