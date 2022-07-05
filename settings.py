class Config():
    def __init__(self, **entries):
        self.__dict__.update(entries)

configs = {
              "roberta-base": {
                  "pretrained_model_type": "bert",
                  "pretrained_model_name": "roberta-base",
                  "lr": 1e-5,
                  "batch_size": 32,
                  "adam_betas": (0.9, 0.98),
                  "weight_decay": 0.01
              },
              "roberta-large": {
                  "pretrained_model_type": "bert",
                  "pretrained_model_name": "roberta-large",
                  "lr": 1e-5,
                  "batch_size": 8,
                  "adam_betas": (0.9, 0.98),
                  "weight_decay": 0.01
              },
              "unifiedqa-base": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-base",
                  "lr": 1e-3,
                  "batch_size": 4,
                  "max_len_in": 512,
                  "max_len_out": 100
              },
              "unifiedqa-large": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-large",
                  "lr": 1e-4,
                  "batch_size": 4,
                  "max_len_in": 256,
                  "max_len_out": 50
              },
              "unifiedqa-large-lr1": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-large",
                  "lr": 1e-5,
                  "batch_size": 4,
                  "max_len_in": 64,
                  "max_len_out": 16
              },
              "unifiedqa-v2-large": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-v2-t5-large-1363200",
                  "lr": 1e-5,
                  "batch_size": 4,
                  "max_len_in": 64,
                  "max_len_out": 16
              },
              "unifiedqa-v2-large-test": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-v2-t5-large-1363200",
                  "lr": 1e-5,
                  "batch_size": 4,
                  "max_len_in": 256,
                  "max_len_out": 50
              },
              "unifiedqa-large-lr2": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-large",
                  "lr": 5e-5,
                  "batch_size": 4,
                  "max_len_in": 256,
                  "max_len_out": 50
              },
              "unifiedqa-large-lr3": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "allenai/unifiedqa-t5-large",
                  "lr": 5e-6,
                  "batch_size": 4,
                  "max_len_in": 256,
                  "max_len_out": 50
              },
              "t5-base": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "t5-base",
                  "lr": 1e-3,
                  "batch_size": 4,
                  "max_len_in": 512,
                  "max_len_out": 100
              },
              "t5-large": {
                  "pretrained_model_type": "t5",
                  "pretrained_model_name": "t5-large",
                  "lr": 1e-5,
                  "batch_size": 4,
                  "max_len_in": 30,
                  "max_len_out": 50
              }
          }

def get_config_by_name(config_name):
    return Config(**configs[config_name])
